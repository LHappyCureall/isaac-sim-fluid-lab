"""Read-only, CPU-only audit of a completed 78-episode V4 experiment.

The completion gate runs before opening any journal, model or heldout trace.
Only the separately named audit JSON is written. Unsafe/invalid episodes are
audited and counted, never removed to improve the reported result.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "assets/robot_testtube_pour/scripts/"
DT, QUANTUM = 1.0 / 60.0, 0.0035 ** 3 * 1e6
TARGETS = (20, 30, 40)
BUDGET = {"training": 48, "validation": 12, "test": 18, "total": 78}
RUNNER_SHA256 = "fe751a2ad2ca2337a35cb0eb5abbaf7d03c649bcd084da6bc73f7536e40cf1b9"
EXPECTED_KIND = "formal_upright_lift_fixed_mouth_CEM_78_live_episodes"


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
        parse_constant=lambda value: (_ for _ in ()).throw(AuditError("Nonfinite JSON: " + value)))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def finite(value):
    if isinstance(value, float):
        need(math.isfinite(value), "Nonfinite recorded number")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def close(a, b, message, tolerance=1e-9):
    need(type(a) in (int, float) and math.isfinite(a) and abs(a - b) <= tolerance,
         f"{message}: {a!r} != {b!r}")


def same(a, b, message):
    if isinstance(b, dict):
        need(isinstance(a, dict) and set(a) == set(b), message + ": keys")
        for key in b:
            same(a[key], b[key], message + "." + str(key))
    elif isinstance(b, list):
        need(isinstance(a, list) and len(a) == len(b), message + ": length")
        for index, (one, two) in enumerate(zip(a, b)):
            same(one, two, f"{message}[{index}]")
    elif isinstance(b, float):
        close(a, b, message, max(1e-9, abs(b) * 2e-12))
    else:
        need(a == b, message)


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "Expected explicit UTC")
    return result


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    need(path.is_relative_to(Path(root).resolve()), "Artifact escapes its run directory")
    return path


def gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            AuditError("Nonfinite JSON: " + value))) for line in stream]
    for row in rows:
        finite(row)
    return rows


def completion_gate(run):
    path = Path(run) / "summary.json"
    need(path.is_file(), "Incomplete experiment: summary.json is absent; no journal/test was read")
    summary = read(path)
    need(summary.get("complete") is True and summary.get("total_recorded_episodes") == 78,
         "Incomplete experiment: complete78 is required before reading any journal/test")
    need(summary.get("budget") == {"generations": 2, "population": 4, "elites": 2, **BUDGET},
         "Unexpected completed budget")
    return summary


def numpy_cpu(root):
    try:
        import numpy
        return numpy
    except ImportError:
        for path in sorted((root / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")):
            sys.path.insert(0, str(path))
        import numpy
        return numpy


class FrozenGeometry:
    """Independent cache polynomials plus raw-URDF FK; imports no runtime."""

    def __init__(self, run, declaration, root):
        self.np = np = numpy_cpu(root)
        entry = declaration["input_files"]["path_cache"]
        with np.load(run / "input_snapshot" / ("path_cache" + Path(entry["path"]).suffix), allow_pickle=False) as source:
            self.data = data = {key: source[key].copy() for key in source.files}
        self.lower = np.asarray(declaration["robot_envelope"]["joint_lower_limits_rad"])
        self.upper = np.asarray(declaration["robot_envelope"]["joint_upper_limits_rad"])
        self.vlimits = np.asarray(declaration["robot_envelope"]["joint_velocity_limits_rad_s"])
        self.vector(data["lower_limits_rad"], self.lower, "Cache lower joint limits")
        self.vector(data["upper_limits_rad"], self.upper, "Cache upper joint limits")
        self.vector(data["velocity_limits_rad_s"], self.vlimits, "Cache joint speed limits")
        urdf_entry = declaration["input_files"]["urdf"]
        urdf = ET.parse(run / "input_snapshot" / ("urdf" + Path(urdf_entry["path"]).suffix)).getroot()
        joints = {item.get("name"): item for item in urdf.findall("joint")}
        names = [str(item) for item in data["joint_names"]]
        self.origins = []
        for index, name in enumerate(names):
            joint = joints[name]
            need(joint.find("axis").get("xyz").split() == ["0", "0", "1"], "Unexpected URDF joint axis")
            for key, values in (("lower", self.lower), ("upper", self.upper), ("velocity", self.vlimits)):
                close(float(joint.find("limit").get(key)), float(values[index]), "Raw URDF limit " + key)
            self.origins.append(self.origin(joint))
        parent = {item.find("child").get("link"): item for item in joints.values()}
        current, end, fixed = "panda_hand", joints[names[-1]].find("child").get("link"), []
        while current != end:
            joint = parent[current]
            need(joint.get("type") == "fixed", "Unexpected movable joint after arm")
            fixed.append(self.origin(joint))
            current = joint.find("parent").get("link")
        self.hand_fixed = np.eye(4)
        for matrix in reversed(fixed):
            self.hand_fixed = self.hand_fixed @ matrix
        self.vector(data["origin_matrices_col"], self.origins, "Raw URDF origins", 3e-7)
        self.vector(data["hand_fixed_matrix_col"], self.hand_fixed, "Raw URDF hand fixed chain", 3e-7)
        self.base = data["base_world_matrix_col"].copy()
        self.ramp = float(data["ramp_s"])
        self.lift_end = float(data["lift_duration_s"])
        self.prepare_end = self.lift_end + float(data["lifted_to_78_duration_s"])
        self.maximum_u = self.prepare_end + float(data["dose_duration_s"])
        self.qhome = data["q_home"].copy()
        need(str(data["base_cache_sha256"]) == declaration["input_files"]["base_path_cache"]["sha256"],
             "Lift-cache base identity differs")

    def vector(self, a, b, message, tolerance=1e-9):
        a, b = self.np.asarray(a), self.np.asarray(b)
        need(a.shape == b.shape and self.np.isfinite(a).all()
             and float(self.np.max(self.np.abs(a - b))) <= tolerance, message)

    def ry(self, theta):
        return self.np.array(((math.cos(theta), 0, math.sin(theta)), (0, 1, 0),
                             (-math.sin(theta), 0, math.cos(theta))))

    def origin(self, joint):
        np, origin = self.np, joint.find("origin")
        xyz = [0., 0., 0.] if origin is None else list(map(float, origin.get("xyz", "0 0 0").split()))
        r, p, y = [0., 0., 0.] if origin is None else list(map(float, origin.get("rpy", "0 0 0").split()))
        rx = np.array(((1, 0, 0), (0, math.cos(r), -math.sin(r)), (0, math.sin(r), math.cos(r))))
        rz = np.array(((math.cos(y), -math.sin(y), 0), (math.sin(y), math.cos(y), 0), (0, 0, 1)))
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rz @ self.ry(p) @ rx, xyz
        return matrix

    def progress(self, t, duration):
        t, r = float(self.np.clip(t, 0, duration)), self.ramp
        peak = 1.0 / (duration - r)
        if t <= r:
            return .5 * peak * (t - r * math.sin(math.pi * t / r) / math.pi)
        if t < duration - r:
            return peak * (t - r / 2)
        left = duration - t
        return 1 - .5 * peak * (left - r * math.sin(math.pi * left / r) / math.pi)

    def polynomial(self, grid, coefficients, value):
        index = int(self.np.clip(self.np.searchsorted(grid, value, side="right") - 1, 0, len(grid) - 2))
        x = (value - grid[index]) / (grid[index + 1] - grid[index])
        return sum(coefficients[index, power] * x ** power for power in range(6))

    def q(self, u):
        d = self.data
        need(-1e-8 <= u <= self.maximum_u + 1e-8, "Path coordinate outside frozen cache")
        u = float(self.np.clip(u, 0, self.maximum_u))
        if u <= self.lift_end:
            return self.polynomial(d["lift_progress_grid"], d["lift_joint_coefficients"], self.progress(u, self.lift_end))
        if u <= self.prepare_end:
            alpha = self.progress(u - self.lift_end, self.prepare_end - self.lift_end)
            return d["q_lift_end"] + (d["theta_joint_q"][0] - d["q_lift_end"]) * alpha
        grid = d["theta_rad"]
        angle = float(grid[0]) + float(grid[-1] - grid[0]) * self.progress(
            u - self.prepare_end, float(d["dose_duration_s"]))
        return self.polynomial(grid, d["theta_joint_coefficients"], angle)

    def hand(self, q):
        np, matrix = self.np, self.base.copy()
        for origin, value in zip(self.origins, q):
            rz = np.eye(4)
            rz[:2, :2] = [[math.cos(value), -math.sin(value)], [math.sin(value), math.cos(value)]]
            matrix = matrix @ origin @ rz
        return matrix @ self.hand_fixed

    def rotation(self, quaternion):
        np = self.np
        x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
        return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def commanded_u(status, start, time_s, geometry):
    elapsed = max(0., time_s - start)
    h = min(elapsed, status["command_horizon_s"])
    if status["phase"] == "prepare":
        return min(status["command_goal_source"], status["u"] + status["actual_u_velocity"] * h)
    if status["phase"] == "home_return" and status.get("home_return_uses_exact_reverse_prepare"):
        return max(0., status["u"] + status["command_source_velocity"] * h)
    velocity, acceleration = status["command_source_velocity"], status["command_acceleration"]
    u = status["u"] + velocity * h + .5 * acceleration * h * h
    velocity += acceleration * h
    if elapsed > status["command_horizon_s"] and velocity:
        extra = min(elapsed - status["command_horizon_s"], abs(velocity) / 1.5)
        u += velocity * extra - .5 * math.copysign(1.5, velocity) * extra * extra
    if status["phase"] == "retract_78" and u <= status["command_goal_source"]:
        u = status["command_goal_source"]
    return min(geometry.maximum_u, max(0., u))


class MotionAudit:
    def __init__(self, geometry):
        self.g, self.np = geometry, geometry.np
        self.dense_q, self.dense_dq, self.dense_time = geometry.qhome.copy(), self.np.zeros(7), 0.
        self.previous_status = self.previous_time = None
        self.dense_count = self.rows = self.reverse_switches = 0
        self.terminal_source_unverifiable_samples = 0
        self.maxima = dict.fromkeys(("joint_speed_limit_ratio", "joint_acceleration_rad_s2", "joint_position_violation_rad",
            "cache_q_residual_rad", "command_u_residual", "actual_body_position_error_m", "actual_body_orientation_error_rad",
            "actual_grasp_error_m", "fixed_mouth_error_m", "pure_Ry_error"), 0.)

    def dense(self, samples, *, terminal=False):
        np, g = self.np, self.g
        for sample in samples:
            t = sample["time_s"]
            if terminal and t <= self.dense_time + 1e-10:
                continue
            close(t - self.dense_time, DT / 4, "240Hz sample interval", 1e-10)
            q = np.asarray(sample["q_rad"])
            violation = float(max(np.max(g.lower - q), np.max(q - g.upper), 0.))
            self.maxima["joint_position_violation_rad"] = max(self.maxima["joint_position_violation_rad"], violation)
            dq = (q - self.dense_q) / (t - self.dense_time)
            ddq = (dq - self.dense_dq) / (t - self.dense_time)
            g.vector(dq, sample["dq_rad_s"], "Independent joint velocity FD", 1e-7)
            g.vector(ddq, sample["ddq_rad_s2"], "Independent joint acceleration FD", 2e-5)
            ratio = float(np.max(np.abs(dq) / g.vlimits))
            close(sample["speed_limit_ratio"], ratio, "Independent joint speed ratio", 1e-7)
            self.maxima["joint_speed_limit_ratio"] = max(self.maxima["joint_speed_limit_ratio"], ratio)
            self.maxima["joint_acceleration_rad_s2"] = max(self.maxima["joint_acceleration_rad_s2"], float(np.max(np.abs(ddq))))
            if self.previous_status is not None:
                u = commanded_u(self.previous_status, self.previous_time, t, g)
                g.vector(q, g.q(u), "Commanded subframe q vs frozen cache", 2e-9)
            elif terminal:
                self.terminal_source_unverifiable_samples += 1
            self.dense_q, self.dense_dq, self.dense_time = q.copy(), dq.copy(), t
            self.dense_count += 1

    def row(self, time_s, kinematics, status):
        np, g = self.np, self.g
        q = np.asarray(kinematics["joint_position_rad"])
        need(q.shape == (7,) and np.isfinite(q).all(), "Seven finite recorded joints required")
        violation = float(max(np.max(g.lower - q), np.max(q - g.upper), 0.))
        self.maxima["joint_position_violation_rad"] = max(self.maxima["joint_position_violation_rad"], violation)
        residual = float(np.max(np.abs(q - g.q(status["u"]))))
        self.maxima["cache_q_residual_rad"] = max(self.maxima["cache_q_residual_rad"], residual)
        need(residual < 2e-9, "Endpoint joints disagree with independent cache evaluation")
        samples = kinematics.get("dense_240_hz_samples", [])
        need(len(samples) == (0 if time_s == 0 else 4), "Completed frame must retain four dense samples")
        self.dense(samples)
        if self.previous_status is not None:
            expected = commanded_u(self.previous_status, self.previous_time, time_s, g)
            residual = abs(expected - status["u"])
            self.maxima["command_u_residual"] = max(self.maxima["command_u_residual"], residual)
            need(residual < 2e-9, "Endpoint source clock differs from prior committed command")
        if abs(status["command_source_velocity"] - status["actual_u_velocity"]) > 1e-8:
            need(status["phase"] == "home_return" and status.get("home_return_uses_exact_reverse_prepare")
                 and abs(status["u"] - g.prepare_end) < 1e-9, "Coordinate-rate jump away from C2 anchor")
            # Both pieces use progress profiles with zero first/second endpoint
            # derivatives. Dense finite differences independently cover it.
            self.reverse_switches += 1
        pose = np.asarray(kinematics["actual_tube_pose_xyzw"])
        need(pose.shape == (7,) and np.isfinite(pose).all(), "Missing actual physical tube pose")
        rotation = g.rotation(pose[3:])
        hand = g.hand(q)
        expected = hand @ g.data["hand_to_tube"]
        pe = float(np.linalg.norm(pose[:3] - expected[:3, 3]))
        re = float(np.arccos(np.clip((np.trace(rotation.T @ expected[:3, :3]) - 1) / 2, -1, 1)))
        grasp = float(np.linalg.norm(hand[:3, 3] + hand[:3, :3] @ np.array([0., 0., .1034])
            - pose[:3] - rotation @ np.array([0., 0., .09])))
        close(kinematics["actual_body_position_error_m"], pe, "Raw URDF/body position error", 5e-8)
        close(kinematics["actual_body_orientation_error_rad"], re, "Raw URDF/body rotation error", 5e-7)
        close(kinematics["actual_grasp_position_error_m"], grasp, "Raw URDF/tube grasp", 5e-7)
        for key, value in (("actual_body_position_error_m", pe), ("actual_body_orientation_error_rad", re), ("actual_grasp_error_m", grasp)):
            self.maxima[key] = max(self.maxima[key], value)
        if status["u"] >= g.prepare_end - 1e-12:
            mouth = pose[:3] + rotation @ np.array([0., 0., .18])
            self.maxima["fixed_mouth_error_m"] = max(self.maxima["fixed_mouth_error_m"], float(np.linalg.norm(mouth - g.data["pivot_m"])))
            theta = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
            self.maxima["pure_Ry_error"] = max(self.maxima["pure_Ry_error"], float(np.linalg.norm(rotation - g.ry(theta))))
        prediction = status["received_ml"] + status["inflight_ml"] + max(0., status["flow_ml_s"]) * status["policy_params"]["tail_lead_s"]
        close(status["policy_action"]["predicted_final_ml"], prediction, "Policy prediction")
        need(status["policy_action"]["stop"] == (prediction + status["policy_params"]["stop_bias_ml"] >= status["target_ml"]), "Policy stop decision")
        self.previous_status, self.previous_time = status, time_s
        self.rows += 1


def partitions(part, truth, target):
    keys = ("beaker", "tube", "airborne", "spilled", "below_table", "tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled")
    need(all(type(part[key]) is int and part[key] >= 0 for key in keys), "Invalid particle count")
    need(part["total"] == 5330 and part["initial_tube"] == 1198
         and sum(part[key] for key in keys[:4]) == 5330
         and sum(part[key] for key in keys[5:]) == 1198, "Particle identity partitions")
    for amount, count in (("received_ml", "tube_to_beaker"), ("remaining_ml", "tube_remaining"),
                          ("inflight_ml", "tube_airborne"), ("spilled_ml", "tube_spilled")):
        close(truth[amount], part[count] * QUANTUM, "Particle-mass volume " + amount)
    if "target_ml" in truth:
        close(truth["target_ml"], target, "Recorded target")


def episode_audit(run, row, declaration, geometry, runner):
    result, conditions = row["result"], row["conditions"]
    same([result[key] for key in ("target_ml", "seed", "randomized", "policy_params")],
         [conditions[key] for key in ("target_ml", "seed", "randomized", "policy_params")], "Condition/footer linkage")
    need(row["conditions_sha256"] == signature(conditions), "Conditions SHA")
    paths = {}
    for key in ("record", "post_record", "motion_record"):
        relative = row[key + "_path"]
        if relative is not None:
            paths[key] = inside(run, relative)
            need(sha(paths[key]) == row[key + "_sha256"], "Raw artifact SHA: " + key)
    action = gz(paths["record"])
    evidence = read(paths["motion_record"])
    same(evidence, row["motion_evidence"], "Motion sidecar/journal")
    runner.record_links(row)  # Secondary frozen-schema check, no mutation/GPU.
    action_check = runner.trace_check(paths["record"], result, declaration)
    same(action_check, row["trace_check"], "Frozen direct trace check")
    steps = [item for item in action if item["type"] == "step"]
    direct = [item for item in action if item["type"] == "direct_step_evidence"]
    failures = [item for item in action if item["type"] == "failure"]
    footer = next(item for item in action if item["type"] == "result")
    same({key: value for key, value in footer.items() if key != "type"}, result, "Unmodified original footer")
    need(len(steps) == len(direct) and bool(steps), "Direct/action row coverage")
    need(all(item["step"] == index and abs(item["physical_time_s"] - index * DT) < 1e-10 for index, item in enumerate(steps)), "True60Hz action clock")
    header, effective = action[0], next(item for item in action if item["type"] == "effective_direct_sources")
    need(header["randomized"] == conditions["randomized"] and header["policy_params"] == conditions["policy_params"], "Raw header actual randomization/policy")
    expected_sources = {Path(declaration["module_binding"][key]).name: declaration["source_sha256"][declaration["module_binding"][key]]
                        for key in ("geometry_file", "controller_file")}
    same(effective["source_sha256"], expected_sources, "Actual imported implementations")
    need(effective["cache_sha256"] == declaration["input_files"]["path_cache"]["sha256"], "Actual cache")
    need(utc(effective["reset_started_utc"]) == utc(row["episode_wall_clock_start_utc"]), "Reset UTC linkage")
    motion, max_spill, max_label0, max_below, mass_drift = MotionAudit(geometry), 0, 0, 0, 0.
    mass = result["actual_initial_total_mass_kg"]
    for step, item in zip(steps, direct):
        need(item["step"] == step["step"] and step["solver_substeps"] >= 4, "Physical step/substep evidence")
        part, truth = item["partitions"], step["truth"]
        partitions(part, truth, result["target_ml"])
        mass_drift = max(mass_drift, abs(truth["mass_kg"] - mass))
        max_spill, max_label0, max_below = max(max_spill, part["spilled"]), max(max_label0, part["spilled"] - part["tube_spilled"]), max(max_below, part["below_table"])
        motion.row(item["physical_time_s"], item["kinematics"], item["controller_status"])
    action_rows, action_dense = motion.rows, motion.dense_count
    for failure in failures:
        motion.dense(failure.get("last_motion_evidence", {}).get("dense_240_hz_samples", []), terminal=True)
    terminal_action_dense = motion.dense_count - action_dense
    action_maxima = dict(motion.maxima)
    status = row["controller_summary"]
    same(status, direct[-1]["controller_status"], "Action-end controller") if not failures else None
    if result["complete"]:
        need(result["physical_time_s"] <= 5 + 1e-10 and result["physical_steps"] <= 300
             and status["done"] and status["completed_by_deadline"] and not status["timed_out"], "Completed5s semantics")
        need(len(direct) >= 13 and all(item["controller_status"]["home_static"] for item in direct[-13:])
             and status["stable_frames"] >= 12 and status["home_quiet_seconds"] >= .2 - 1e-9, "Home12frames/quiet")
    post_path = paths.get("post_record")
    post_rows = gz(post_path) if post_path else []
    post = row["post_observation"]
    post_error = bool(isinstance(post, dict) and post.get("error"))
    post_check = runner.ref.verify_post_trace(post_path, post, result)
    same(post_check, row["post_trace_check"], "Frozen post-clock check")
    deadline_received = result["received_ml"] if abs(result["physical_time_s"] - 5) < 1e-10 else None
    max_change, post_last_time = 0., result["physical_time_s"]
    for index, item in enumerate(post_rows, 1):
        t, part = item["physical_time_s"], item["measurement"]
        need(item["step"] == index and item["solver_substeps"] >= 4, "Post step/substeps")
        close(t, result["physical_time_s"] + index * DT, "True60Hz post clock", 1e-10)
        partitions(part, part, result["target_ml"])
        need(part["volume_transform_basis"] == "actual_solver_body_q_full_R", "Post volume basis")
        mass_drift = max(mass_drift, abs(part["total_mass_kg"] - mass))
        max_spill, max_label0, max_below = max(max_spill, part["spilled"]), max(max_label0, part["spilled"] - part["tube_spilled"]), max(max_below, part["below_table"])
        max_change = max(max_change, abs(part["received_ml"] - result["received_ml"]))
        if abs(t - 5) < 1e-10:
            deadline_received = part["received_ml"]
        motion.row(t, item["kinematics"], item["controller_status"])
        post_last_time = t
    if isinstance(post, dict) and not post_error:
        need(post_last_time >= 7 - 1e-10 and len(post_rows) == round((max(7., result["physical_time_s"] + 2) - result["physical_time_s"]) / DT), "Complete post-to7 observation")
        close(post["start_physical_time_s"], result["physical_time_s"], "Post start")
        close(post["deadline_received_ml"], deadline_received, "Exact5s dose")
        close(post["maximum_received_change_ml"], max_change, "Observed post dose change")
    terminal_post = evidence["post_terminal_observation"]
    need(bool(terminal_post["unlogged_terminal_post_frame_possible"]) == post_error, "Post exception evidence flag")
    terminal = terminal_post["kinematics"]
    terminal_dense_before = motion.dense_count
    if post_error:
        motion.dense(terminal.get("dense_240_hz_samples", []), terminal=True)
    if result["mass_conserved"]:
        need(mass_drift <= 1e-12 and result["particle_mass_array_unchanged"] and result["mass_volume_units_consistent"], "Mass-conserved footer differs from logged sums")
    if failures:
        need(not result["valid_physics"], "Failure was marked valid")
        need(result["measurement_basis"] in ("liquid_measurement_at_reported_sample_time", "last_complete_measurement_before_partial_fluid_failure"), "Unknown failure measurement basis")
        for failure in failures:
            need("failure_stage" in failure and "measurement_physical_time_s" in failure and "last_motion_evidence" in failure, "Missing failed-frame basis/evidence")
    assessment = runner.assess(result, status, row["motion_metrics"], post, action_check, post_check, evidence["post_motion_metrics"])
    same(assessment, row["assessment"], "Assessment reconstruction")
    if assessment["physical_safe"]:
        need(max_spill == max_below == 0 and motion.maxima["joint_speed_limit_ratio"] <= 1.00001
             and motion.maxima["joint_acceleration_rad_s2"] <= 25.0001
             and motion.maxima["joint_position_violation_rad"] <= 1e-6
             and motion.maxima["actual_body_position_error_m"] <= 1e-5
             and motion.maxima["actual_body_orientation_error_rad"] <= 1e-4, "Independent safety contradicts claimed safe episode")
    return {"episode_id": row["episode_id"], "target_ml": result["target_ml"], "seed": result["seed"],
        "stage": conditions["stage"], "policy_role": conditions["policy_role"], "action_rows": action_rows,
        "action_dense_240hz_rows": action_dense, "post_rows": len(post_rows), "total_dense_240hz_rows": motion.dense_count,
        "independent_motion_maxima": motion.maxima, "action_motion_maxima": action_maxima,
        "maximum_logged_mass_drift_kg": mass_drift, "maximum_total_spill_particles": max_spill,
        "maximum_initial_beaker_label_spill_particles": max_label0, "maximum_below_table_particles": max_below,
        "deadline_received_ml": deadline_received, "post_final_received_ml": post_rows[-1]["measurement"]["received_ml"] if post_rows else None,
        "post_maximum_dose_change_ml": max_change, "post_end_logged_time_s": post_last_time,
        "action_failure_events": failures, "action_unlogged_terminal_frames": action_check["unlogged_terminal_failure_steps"],
        "measurement_basis": result["measurement_basis"], "measurement_physical_time_s": result["measurement_physical_time_s"],
        "post_exception": post_error, "additional_terminal_action_dense_rows_audited": terminal_action_dense,
        "additional_terminal_post_dense_rows_audited": motion.dense_count - terminal_dense_before,
        "assessment": assessment, "source_coordinate_C2_reverse_switches": motion.reverse_switches,
        "terminal_commanded_source_unverifiable_samples": motion.terminal_source_unverifiable_samples,
        "original_result_unchanged": True}


def metrics(records):
    need(bool(records), "Empty metrics group")
    n, errors = len(records), [item["assessment"]["absolute_error_ml"] for item in records]
    return {"episodes": n, "mae_ml": sum(errors) / n, "worst_absolute_error_ml": max(errors),
        "unsafe_count": sum(not item["assessment"]["physical_safe"] for item in records),
        "deadline_or_home_or_late_drop_failures": sum(not (item["assessment"]["completed_by_deadline"] and item["assessment"]["late_drop_passed"]) for item in records),
        "feasible_count": sum(item["assessment"]["constraint_feasible"] for item in records),
        "missing_deadline_measurements": sum(not item["assessment"]["deadline_dose_measurement_available"] for item in records),
        "within_0_1_ml_count": sum(item["assessment"]["success_within_0_1_ml"] for item in records),
        "within_0_2_ml_count": sum(item["assessment"]["constraint_feasible"] and error <= .2 for item, error in zip(records, errors)),
        "all_mass_conserved": all(item["result"]["mass_conserved"] for item in records),
        "mean_action_physical_seconds": sum(item["result"]["physical_time_s"] for item in records) / n,
        "physical_steps": sum(item["result"]["physical_steps"] for item in records),
        "action_physical_seconds": sum(item["result"]["physical_time_s"] for item in records),
        "post_observation_physical_seconds": sum(item["post_observation"].get("duration_s", 0) for item in records if isinstance(item["post_observation"], dict))}


def key(records):
    overall = metrics(records)
    groups = [records] + [[item for item in records if item["conditions"]["target_ml"] == target] for target in TARGETS]
    violation = sum(max(0., metrics(group)["mae_ml"] - .1) + max(0., metrics(group)["worst_absolute_error_ml"] - .2) for group in groups if group)
    return (overall["unsafe_count"], overall["deadline_or_home_or_late_drop_failures"], violation,
            overall["mae_ml"] + .5 * overall["worst_absolute_error_ml"], overall["mean_action_physical_seconds"])


def release(records):
    groups = [records] + [[item for item in records if item["conditions"]["target_ml"] == target] for target in TARGETS]
    return bool(all(groups) and all(metrics(group)["unsafe_count"] == metrics(group)["deadline_or_home_or_late_drop_failures"] == 0
        and metrics(group)["mae_ml"] <= .1 and metrics(group)["worst_absolute_error_ml"] <= .2
        and metrics(group)["all_mass_conserved"] for group in groups))


class ReplayCEM:
    def __init__(self, declaration):
        c = declaration["cem"]
        self.names, self.bounds, self.mean = c["parameter_names"], c["bounds"], dict(c["initial_params"])
        self.std = {name: (self.bounds[name][1] - self.bounds[name][0]) * c["initial_std_fraction"] for name in self.names}
        self.rng, self.best, self.best_key = random.Random(c["seed"]), None, None

    def ask(self):
        result = [dict(self.mean)]
        if self.best is not None and self.best != self.mean:
            result.append(dict(self.best))
        while len(result) < 4:
            result.append({name: min(self.bounds[name][1], max(self.bounds[name][0], self.rng.gauss(self.mean[name], self.std[name]))) for name in self.names})
        return result

    def tell(self, parameters, keys):
        order = sorted(range(4), key=lambda index: (tuple(keys[index]), index))
        for index in order:
            if self.best_key is None or keys[index] < self.best_key:
                self.best_key, self.best = keys[index], dict(parameters[index])
        elites = [parameters[index] for index in order[:2]]
        for name in self.names:
            average = sum(item[name] for item in elites) / 2
            deviation = math.sqrt(sum((item[name] - average) ** 2 for item in elites) / 2)
            self.mean[name] = .65 * self.mean[name] + .35 * average
            self.std[name] = max(.02 * (self.bounds[name][1] - self.bounds[name][0]), .65 * self.std[name] + .35 * deviation)
        return {"elite_indices": order[:2], "best_key": list(self.best_key), "best_params": dict(self.best),
                "next_mean": dict(self.mean), "next_std": dict(self.std)}


def expected_order(declaration):
    order = [f"train_g{g:03d}_c{c:03d}_t{target}_s{seed}" for g in range(2) for c in range(4) for target in TARGETS for seed in declaration["train_seeds"]]
    order += [f"validation_{role}_t{target}_s{seed}" for role in ("candidate", "baseline") for seed in declaration["validation_seeds"] for target in TARGETS]
    order += [f"test_{role}_t{target}_s{seed}" for seed in declaration["test_seeds"] for target in TARGETS for role in ("candidate", "baseline")]
    return order


def verify_inputs(run, root, declaration, protocol):
    need(len(declaration["source_sha256"]) == 30 and len(declaration["input_files"]) == 15, "Expected30 sources and15 frozen inputs")
    for name, digest in declaration["source_sha256"].items():
        need(sha(root / name) == sha(run / "source_snapshot" / name) == digest, "Native/source snapshot SHA: " + name)
    for role, entry in declaration["input_files"].items():
        snap = run / "input_snapshot" / (role + Path(entry["path"]).suffix)
        need(sha(entry["path"]) == sha(snap) == entry["sha256"], "Native/input snapshot SHA: " + role)
    frozen = read(run / "input_freeze.json")
    need(frozen["declaration"] == declaration and frozen["input_id"] == signature(declaration) == protocol["input_id"], "Canonical input identity")
    legacy = read(run / "input_snapshot/protected_legacy_protocol.json")["protocol"]["source_sha256"]
    need(len(legacy) == 20 and all(declaration["source_sha256"].get(name) == digest for name, digest in legacy.items()), "Protected original20 sources")
    gui = {path.relative_to(root).as_posix(): sha(path) for path in sorted((root / "assets/robot_testtube_pour/policy").glob("*.json"))}
    need(gui == declaration["gui_checkpoint_sha256_before"], "Original GUI model set/bytes changed")
    for name, digest in declaration["historical_protocol_sha256"].items():
        need(sha(root / name) == digest, "Pinned historical protocol changed")
    train, val, test = map(set, (declaration["train_seeds"], declaration["validation_seeds"], declaration["test_seeds"]))
    need(train == {81101, 81102} and val == {82101, 82102} and test == {91101, 91102, 91103}
         and not train & val and not train & test and not val & test, "Fresh declared seed partitions")
    def seeds(value):
        found = set()
        if isinstance(value, dict):
            for name, child in value.items():
                if name == "seed" and type(child) is int:
                    found.add(child)
                elif name.endswith("seeds") and isinstance(child, list):
                    found.update(item for item in child if type(item) is int)
                found.update(seeds(child))
        elif isinstance(value, list):
            for child in value:
                found.update(seeds(child))
        return found
    proposed = train | val | test | {declaration["cem"]["seed"]}
    for name in declaration["historical_protocol_sha256"]:
        need(not proposed.intersection(seeds(read(root / name))), "A proposed seed exists in a pinned historical protocol")
    need(declaration["automatic_gui_deployment"] is False and declaration["neural_policy"] is False, "Unexpected deployment/training claim")
    return gui


def audit(run, root):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary = completion_gate(run)  # Must remain the first experiment read.
    document = read(run / "protocol.json")
    protocol, pid = document["protocol"], document["protocol_id"]
    need(protocol["kind"] == EXPECTED_KIND and signature(protocol) == pid == summary["protocol_id"], "Canonical protocol identity")
    declaration = protocol["declaration"]
    same(summary["protocol"], protocol, "Summary protocol")
    gui = verify_inputs(run, root, declaration, protocol)
    runner_path = root / "train_target_volume_pivot.py"
    need(sha(runner_path) == RUNNER_SHA256, "Frozen runner source is not the audited implementation")
    spec = importlib.util.spec_from_file_location("completed_pivot_frozen_schema_helpers", runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)  # Definitions/pure I/O only; no SDK init.
    journal_path = run / "episodes.jsonl"
    records = {}
    order = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        finite(row)
        name = row["episode_id"]
        need(name not in records and row["protocol_id"] == pid, "Duplicate/foreign journal entry")
        records[name], order = row, order + [name]
    need(order == expected_order(declaration) and len(records) == 78, "Exactly48/12/18 unique scheduled episodes in order required")
    stages = Counter(row["conditions"]["stage"] for row in records.values())
    need(stages == {"train": 48, "validation": 12, "test": 18}, "Stage budgets")
    geometry = FrozenGeometry(run, declaration, root)
    episode_reports = [episode_audit(run, records[name], declaration, geometry, runner) for name in order]
    replay, training = ReplayCEM(declaration), []
    for generation in range(2):
        parameters = replay.ask()
        groups = []
        for index, params in enumerate(parameters):
            group = [records[f"train_g{generation:03d}_c{index:03d}_t{target}_s{seed}"] for target in TARGETS for seed in declaration["train_seeds"]]
            for row in group:
                need(row["conditions"]["policy_params"] == params and row["conditions"]["generation"] == generation and row["conditions"]["candidate"] == index, "Independent CEM scheduled parameters")
            groups.append(group)
            training.extend(group)
        keys = [key(group) for group in groups]
        update = replay.tell(parameters, keys)
        expected = {"generation": generation, "candidates": [{"index": index, "params": params, "selection_key": list(score),
            "metrics": metrics(group), "episode_ids": [row["episode_id"] for row in group]} for index, (params, score, group) in enumerate(zip(parameters, keys, groups))],
            "completed_training_episodes": len(training), **update}
        same(summary["training"]["generations"][generation], expected, "Independent CEM generation")
    candidate, baseline = replay.best, declaration["cem"]["baseline_params"]
    same(summary["training"]["best_params"], candidate, "Training candidate")
    validation = {role: [records[f"validation_{role}_t{target}_s{seed}"] for seed in declaration["validation_seeds"] for target in TARGETS] for role in ("candidate", "baseline")}
    recommendation = "candidate" if candidate != baseline and key(validation["candidate"]) < key(validation["baseline"]) else "baseline"
    need(summary["validation"]["recommended_role"] == recommendation and summary["recommendation"]["role"] == recommendation,
         "Validation-only selection was changed")
    freeze = read(run / "freeze.json")
    need(freeze["recommended_role"] == recommendation and freeze["test_used_for_selection"] is False, "Pre-test role freeze")
    frozen_at = utc(freeze["frozen_at_utc"])
    for role, params in (("candidate", candidate), ("recommended", candidate if recommendation == "candidate" else baseline)):
        path = run / (role + "_frozen.json")
        model = read(path)
        need(sha(path) == freeze[role + "_sha256"] == summary["models"][role]["sha256"], "Frozen model bytes")
        same(model["parameters"], params, "Frozen model parameters")
        need(model["metadata"]["source_sha256"] == declaration["source_sha256"] and model["metadata"]["frozen_at_utc"] == freeze["frozen_at_utc"], "Frozen model provenance/time")
    tests = {role: [records[f"test_{role}_t{target}_s{seed}"] for seed in declaration["test_seeds"] for target in TARGETS] for role in ("candidate", "baseline")}
    first_test = min(utc(row["episode_wall_clock_start_utc"]) for rows in tests.values() for row in rows)
    need(first_test > frozen_at, "A physical heldout reset preceded the model freeze")
    test_metrics = {role: metrics(rows) for role, rows in tests.items()}
    for role, rows in tests.items():
        same(summary["test"]["metrics"][role], test_metrics[role], "Independent heldout metrics")
        need(summary["release_acceptance"][role + "_passed"] == release(rows), "Independent release gate")
    runner.verify_complete(run, summary, pid, declaration, records)  # Secondary entire frozen-schema replay.
    declared_paths = {row[key] for row in records.values() for key in ("record_path", "post_record_path", "motion_record_path") if row[key] is not None}
    orphan_completed = []
    for path in (run / "traces").glob("*.steps.jsonl.gz"):
        if path.relative_to(run).as_posix() not in declared_paths:
            rows = gz(path)
            if any(row.get("type") == "result" for row in rows):
                orphan_completed.append(path.name)
    need(not orphan_completed, "Uncommitted completed actions would hide extra physical executions")
    interrupted = []
    interrupted_path = run / "interrupted_artifacts.jsonl"
    if interrupted_path.is_file():
        for line in interrupted_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            path = inside(run, entry["preserved_path"])
            need(sha(path) == entry["sha256"], "Interrupted attempt bytes were not retained")
            interrupted.append(entry)
    need(len(interrupted) == summary["interrupted_artifact_records"], "Interrupted-attempt record count")
    job = read(run / "job_state.json") if (run / "job_state.json").is_file() else None
    if job:
        normalize = lambda mapping: {name.replace("\\", "/"): digest for name, digest in mapping.items()}
        need(normalize(job["gui_checkpoint_sha256_before"]) == gui and job["source_sha256"] == declaration["source_sha256"], "Supervisor source/GUI identities")
    need(summary["gui_checkpoint_sha256_after"] == gui and summary["automatic_gui_deployment"] is False, "GUI changed or deployment occurred")
    all_reports = {row["episode_id"]: row for row in episode_reports}
    per_target = [{"target_ml": target, **{role: metrics([row for row in tests[role] if row["conditions"]["target_ml"] == target]) for role in tests}} for target in TARGETS]
    return {"audit_schema_version": 1, "kind": "independent_completed_pivot_v4_audit", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_tool_sha256": sha(__file__), "artifact_integrity_verified": True, "all_failures_retained": True,
        "protocol_id": pid, "input_id": protocol["input_id"], "summary_sha256": sha(run / "summary.json"), "journal_sha256": sha(journal_path),
        "budgets": BUDGET, "episodes_audited": 78, "raw_action_trace_count": 78,
        "raw_post_trace_count": sum(row["post_record_path"] is not None for row in records.values()),
        "source_files_and_snapshots_verified": 30, "input_files_and_snapshots_verified": 15,
        "protected_legacy_sources_verified": 20, "GUI_models_unchanged": True,
        "raw_action_result_footers_unchanged": True, "CEM_replay_verified": True, "validation_only_recommendation_verified": True,
        "recommended_role": recommendation, "freeze_to_first_test_reset_seconds": (first_test - frozen_at).total_seconds(),
        "test_metrics": test_metrics, "per_target_comparison": per_target, "release_acceptance": summary["release_acceptance"],
        "joint_240hz_samples_audited": sum(row["total_dense_240hz_rows"] for row in episode_reports),
        "FK_full_R_60hz_rows_audited": sum(row["action_rows"] + row["post_rows"] for row in episode_reports),
        "episode_audits": all_reports,
        "all_original_failed_case_ids": [row["episode_id"] for row in records.values() if not row["result"]["success"]],
        "all_assessment_0_1_failure_ids": [row["episode_id"] for row in records.values() if not row["assessment"]["success_within_0_1_ml"]],
        "physical_constraint_failure_ids": [row["episode_id"] for row in records.values() if not row["assessment"]["constraint_feasible"]],
        "interrupted_artifacts_verified": interrupted,
        "task_release_success": summary["release_acceptance"]["recommended_passed"], "no_liquid_or_GPU_execution": True,
        "limitations": ["Particle-center mass counting is privileged simulation truth, not camera/scale or real-hardware accuracy.",
            "Raw traces store aggregate mass/partition evidence, not full per-particle arrays. Full-state finite and unchanged mass-array assertions come from the frozen live collector; aggregate evidence is independently checked.",
            "240Hz q/dq/ddq and60Hz raw-URDF FK/full-R checks are sampled and do not establish torque, friction-grasp or continuous collision safety.25rad/s2 is a simulation envelope.",
            "Recorded UTC reset/freeze order is from this host. Frozen source scheduling and both model-byte hashes support freeze-before-test; file modification times are not used as proof.",
            "A failed/missing5s sample remains invalid; terminal fallback errors are descriptive only. Unsafe/timeout/partial-frame evidence is retained and never rescued by post observation.",
            "Only two training, two validation and three heldout seeds with small position/viscosity variation were used. Completed budget alone is not task success."]}


def self_test():
    # No real run/journal/test is opened. A partial experiment must fail before
    # looking for a protocol, and malformed partition/clock fixtures must fail.
    checks = []
    with tempfile.TemporaryDirectory(prefix="pivot_audit_cpu_", dir=Path(__file__).parent) as temporary:
        folder = Path(temporary)
        for value in (None, {"complete": False, "total_recorded_episodes": 78}, {"complete": True, "total_recorded_episodes": 77}):
            if value is not None:
                (folder / "summary.json").write_text(json.dumps(value), encoding="utf-8")
            try:
                completion_gate(folder)
            except AuditError:
                checks.append("partial_gate_rejected_without_journal")
            else:
                raise AuditError("Partial fixture incorrectly passed")
        part = {"total": 5330, "initial_tube": 1198, "beaker": 4132, "tube": 1198, "airborne": 0, "spilled": 0,
                "below_table": 0, "tube_to_beaker": 0, "tube_remaining": 1198, "tube_airborne": 0, "tube_spilled": 0}
        truth = {"received_ml": 0., "remaining_ml": 1198 * QUANTUM, "inflight_ml": 0., "spilled_ml": 0.}
        partitions(part, truth, 20)
        try:
            partitions({**part, "tube_remaining": 1197}, truth, 20)
        except AuditError:
            checks.append("broken_source_partition_rejected")
        else:
            raise AuditError("Broken partition passed")
        try:
            partitions(part, {**truth, "received_ml": 1.}, 20)
        except AuditError:
            checks.append("wrong_mass_volume_rejected")
        else:
            raise AuditError("Wrong volume passed")
    return {"self_test_passed": True, "checks": checks, "no_real_journal_or_test_read": True, "no_GPU": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "droplet_out/pour_policy_pivot_v4/run_20261122_joint_cem")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    report = audit(args.run, args.root)
    output = args.output or args.run / "pivot_run_audit.json"
    if output.exists():
        previous = read(output)
        need(previous.get("kind") == report["kind"] and previous.get("protocol_id") == report["protocol_id"], "Refusing to replace an unrelated artifact")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_file": str(output), "sha256": sha(output), "episodes_audited": 78,
        "artifact_integrity_verified": True, "recommended_role": report["recommended_role"],
        "task_release_success": report["task_release_success"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print("Audit refused: " + str(error), file=sys.stderr)
        raise SystemExit(2)
