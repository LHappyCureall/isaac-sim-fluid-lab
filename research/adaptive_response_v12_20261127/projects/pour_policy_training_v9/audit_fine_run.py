"""Independent CPU audit of a COMPLETE42-case V9 fine-particle experiment.

The summary gate precedes all journal/model/heldout reads. Runtime modules,
Kit, Warp and CUDA are never imported. Failed cases remain in the audit.
Only the explicitly named final audit output is written by the normal CLI.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_fine_v9/run_20261127"
DT, QUANTUM = 1 / 60, .0027 ** 3 * 1e6
TARGETS, TRAIN, VAL, TEST = (20, 30, 40), (171101,), (172101,), (173101, 173102, 173103)
BUDGET = {"train": 18, "validation": 6, "test": 18, "total": 42}
PREFIX = "assets/robot_testtube_pour/scripts/"
V9 = "projects/pour_policy_training_v9/"
URDF = "exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
URDF_SHA256 = "5aece94c95106e722825bc0487ec9ba3b1162a202c00f38a936a7a18523e8c92"
RUNNER_SHA256 = "8a9f10e4d2448ff67ab44e5e6bf58c01ecaffd2d8b3a4441249403025dcdb251"


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_constant(value):
    raise AuditError("Nonfinite JSON value: " + value)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, float):
        need(math.isfinite(value), "Nonfinite recorded value")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def close(one, two, message, tolerance=1e-9):
    need(type(one) in (int, float) and math.isfinite(one) and abs(one - two) <= tolerance,
         f"{message}: {one!r} != {two!r}")


def same(one, two, message):
    if isinstance(two, dict):
        need(isinstance(one, dict) and set(one) == set(two), message + ": keys")
        for key in two:
            same(one[key], two[key], message + "." + str(key))
    elif isinstance(two, (list, tuple)):
        need(isinstance(one, (list, tuple)) and len(one) == len(two), message + ": length")
        for index, (a, b) in enumerate(zip(one, two)):
            same(a, b, f"{message}[{index}]")
    elif isinstance(two, float):
        close(one, two, message, max(1e-9, abs(two) * 2e-12))
    else:
        need(one == two, message)


def inside(root, relative):
    need(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative artifact path required")
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), "Artifact path escapes its root")
    return path


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "Explicit UTC required")
    return parsed


def completion_gate(run):
    path = run / "summary.json"
    need(path.is_file(), "No final summary; journal/model/test not opened")
    summary = read(path)
    need(summary.get("complete") is True and summary.get("counts") == {k: BUDGET[k] for k in ("train", "validation", "test")},
         "A complete18/6/18 experiment is required before journal/model/test reads")
    return summary


def gz(path, allow_partial=False):
    rows, error = [], None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line, parse_constant=reject_constant)
                finite(row)
                rows.append(row)
    except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        need(allow_partial, "Truncated valid trace: " + str(path))
        error = repr(exc)
    return rows, error


def numpy_cpu(root):
    try:
        import numpy
    except ImportError:
        for path in sorted((root / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")):
            sys.path.insert(0, str(path))
        import numpy
    return numpy


class Geometry:
    """Independent polynomial clock and raw URDF FK; imports no runtime."""

    def __init__(self, run, root, plan):
        self.np = np = numpy_cpu(root)
        selected = "projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz"
        need(selected in plan["input_sha256"], "Selected path cache must be frozen")
        with np.load(inside(run / "input_snapshot", selected), allow_pickle=False) as archive:
            self.data = d = {key: archive[key].copy() for key in archive.files}
        for value in d.values():
            if value.dtype.kind in "fci":
                need(np.isfinite(value).all(), "Nonfinite path cache")
        self.lower, self.upper, self.limits = d["lower_limits_rad"], d["upper_limits_rad"], d["velocity_limits_rad_s"]
        self.qhome, self.base = d["q_home"], d["base_world_matrix_col"]
        self.ramp, self.lift = float(d["ramp_s"]), float(d["lift_duration_s"])
        self.prepare = self.lift + float(d["lifted_to_78_duration_s"])
        self.maximum = self.prepare + float(d["dose_duration_s"])
        self.vector(d["pivot_m"], [.43, .1, .22], "Fixed mouth pivot")
        # The original SDK URDF is an explicit independent audit reference,
        # not silently counted as a seventh frozen experiment input.
        urdf_path = root / URDF
        need(sha(urdf_path) == URDF_SHA256, "Independent reference URDF changed or is unavailable")
        document = ET.parse(urdf_path).getroot()
        joints = {item.get("name"): item for item in document.findall("joint")}
        names = [str(item) for item in d["joint_names"]]
        self.origins = []
        for index, name in enumerate(names):
            joint = joints[name]
            need(joint.find("axis").get("xyz").split() == ["0", "0", "1"], "Unexpected joint axis")
            for key, values in (("lower", self.lower), ("upper", self.upper), ("velocity", self.limits)):
                close(float(joint.find("limit").get(key)), float(values[index]), "Raw URDF limit")
            self.origins.append(self.origin(joint))
        parent = {item.find("child").get("link"): item for item in joints.values()}
        link, arm_end, fixed = "panda_hand", joints[names[-1]].find("child").get("link"), []
        while link != arm_end:
            joint = parent[link]
            need(joint.get("type") == "fixed", "Unexpected nonfixed hand chain")
            fixed.append(self.origin(joint))
            link = joint.find("parent").get("link")
        self.hand_fixed = np.eye(4)
        for matrix in reversed(fixed):
            self.hand_fixed = self.hand_fixed @ matrix
        self.vector(d["origin_matrices_col"], self.origins, "Cache vs raw URDF origins", 3e-7)
        self.vector(d["hand_fixed_matrix_col"], self.hand_fixed, "Cache vs raw URDF hand chain", 3e-7)

    def vector(self, one, two, message, tolerance=1e-9):
        a, b = self.np.asarray(one), self.np.asarray(two)
        need(a.shape == b.shape and self.np.isfinite(a).all()
             and float(self.np.max(self.np.abs(a - b))) <= tolerance, message)

    def ry(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return self.np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    def origin(self, joint):
        np, source = self.np, joint.find("origin")
        xyz = [0., 0., 0.] if source is None else list(map(float, source.get("xyz", "0 0 0").split()))
        r, p, y = [0., 0., 0.] if source is None else list(map(float, source.get("rpy", "0 0 0").split()))
        rx = np.array(((1, 0, 0), (0, math.cos(r), -math.sin(r)), (0, math.sin(r), math.cos(r))))
        rz = np.array(((math.cos(y), -math.sin(y), 0), (math.sin(y), math.cos(y), 0), (0, 0, 1)))
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = rz @ self.ry(p) @ rx, xyz
        return result

    def progress(self, t, duration):
        t, r = float(self.np.clip(t, 0, duration)), self.ramp
        peak = 1 / (duration - r)
        if t <= r:
            return .5 * peak * (t - r * math.sin(math.pi * t / r) / math.pi)
        if t < duration - r:
            return peak * (t - r / 2)
        left = duration - t
        return 1 - .5 * peak * (left - r * math.sin(math.pi * left / r) / math.pi)

    def polynomial(self, grid, coeff, value):
        index = int(self.np.clip(self.np.searchsorted(grid, value, side="right") - 1, 0, len(grid) - 2))
        x = (value - grid[index]) / (grid[index + 1] - grid[index])
        return sum(coeff[index, power] * x ** power for power in range(6))

    def q(self, u):
        need(-1e-8 <= u <= self.maximum + 1e-8, "Coordinate outside frozen path")
        u, d = float(self.np.clip(u, 0, self.maximum)), self.data
        if u <= self.lift:
            return self.polynomial(d["lift_progress_grid"], d["lift_joint_coefficients"], self.progress(u, self.lift))
        if u <= self.prepare:
            return d["q_lift_end"] + (d["theta_joint_q"][0] - d["q_lift_end"]) * self.progress(u - self.lift, self.prepare - self.lift)
        angles = d["theta_rad"]
        angle = float(angles[0]) + float(angles[-1] - angles[0]) * self.progress(u - self.prepare, float(d["dose_duration_s"]))
        return self.polynomial(angles, d["theta_joint_coefficients"], angle)

    def hand(self, q):
        np, matrix = self.np, self.base.copy()
        for origin, angle in zip(self.origins, q):
            rz = np.eye(4)
            rz[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            matrix = matrix @ origin @ rz
        return matrix @ self.hand_fixed

    def rotation(self, quaternion):
        np = self.np
        need(np.linalg.norm(quaternion) > 1e-12, "Zero physical quaternion")
        x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
        return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def commanded_u(status, u, start, time_s, geometry):
    elapsed = max(0., time_s - start)
    h = min(elapsed, status["command_horizon_s"])
    if status["phase"] == "prepare":
        return min(status["command_goal_source"], u + status["command_source_velocity"] * h)
    if status["phase"] == "home_return" and status.get("motion_continuity", "").startswith("C2 physical reverse prepare clock"):
        return max(0., u + status["command_source_velocity"] * h)
    velocity, acceleration = status["command_source_velocity"], status["command_acceleration"]
    value = u + velocity * h + .5 * acceleration * h * h
    velocity += acceleration * h
    if elapsed > status["command_horizon_s"] and velocity:
        extra = min(elapsed - status["command_horizon_s"], abs(velocity) / 1.5)
        value += velocity * extra - .5 * math.copysign(1.5, velocity) * extra * extra
    if status["phase"] == "retract_78" and value <= status["command_goal_source"]:
        value = status["command_goal_source"]
    return min(geometry.maximum, max(0., value))


class Motion:
    def __init__(self, geometry):
        self.g, self.np = geometry, geometry.np
        self.last_q, self.last_dq, self.last_t = geometry.qhome.copy(), self.np.zeros(7), 0.
        self.previous = None
        self.rows = self.dense_rows = 0
        self.maximum = dict.fromkeys(("speed_limit_ratio", "acceleration_rad_s2", "position_violation_rad",
            "body_position_error_m", "body_orientation_error_rad", "grasp_error_m", "dose_mouth_error_m"), 0.)

    def dense(self, samples, terminal=False):
        np, g = self.np, self.g
        for sample in samples:
            t = sample["time_s"]
            if terminal and t <= self.last_t + 1e-10:
                continue
            close(t - self.last_t, DT / 4, "240Hz clock", 1e-9)
            q = np.asarray(sample["q_rad"])
            need(q.shape == (7,) and np.isfinite(q).all(), "Seven finite dense joints required")
            dq = (q - self.last_q) / (t - self.last_t)
            ddq = (dq - self.last_dq) / (t - self.last_t)
            g.vector(dq, sample["dq_rad_s"], "Independent qdot240", 1e-7)
            g.vector(ddq, sample["ddq_rad_s2"], "Independent qddot240", 2e-5)
            ratio = float(np.max(np.abs(dq) / g.limits))
            close(sample["speed_limit_ratio"], ratio, "Independent speed ratio", 1e-7)
            self.maximum["speed_limit_ratio"] = max(self.maximum["speed_limit_ratio"], ratio)
            self.maximum["acceleration_rad_s2"] = max(self.maximum["acceleration_rad_s2"], float(np.max(np.abs(ddq))))
            self.maximum["position_violation_rad"] = max(self.maximum["position_violation_rad"], float(max(np.max(g.lower - q), np.max(q - g.upper), 0.)))
            if self.previous is not None:
                status, u, start = self.previous
                g.vector(q, g.q(commanded_u(status, u, start, t, g)), "Committed source to independent dense q", 2e-9)
            self.last_q, self.last_dq, self.last_t = q.copy(), dq.copy(), t
            self.dense_rows += 1

    def row(self, t, u, kin, status):
        np, g = self.np, self.g
        q = np.asarray(kin["joint_position_rad"])
        g.vector(q, g.q(u), "Endpoint source to independent q", 2e-9)
        samples = kin.get("dense_240_hz_samples", [])
        need(len(samples) == (0 if t == 0 else 4), "Four actual dense samples per complete frame")
        self.dense(samples)
        if self.previous is not None:
            prior, prior_u, start = self.previous
            close(u, commanded_u(prior, prior_u, start, t, g), "Committed source clock", 2e-9)
        pose = np.asarray(kin["actual_tube_pose_xyzw"])
        need(pose.shape == (7,) and np.isfinite(pose).all(), "Actual full-R physical tube pose required")
        rotation = g.rotation(pose[3:])
        hand = g.hand(q)
        expected = hand @ g.data["hand_to_tube"]
        position = float(np.linalg.norm(pose[:3] - expected[:3, 3]))
        orientation = float(np.arccos(np.clip((np.trace(rotation.T @ expected[:3, :3]) - 1) / 2, -1, 1)))
        grasp = float(np.linalg.norm(hand[:3, 3] + hand[:3, :3] @ np.array([0., 0., .1034])
                                     - pose[:3] - rotation @ np.array([0., 0., .09])))
        close(kin["actual_body_position_error_m"], position, "Independent raw URDF/body position", 5e-8)
        close(kin["actual_body_orientation_error_rad"], orientation, "Independent raw URDF/body orientation", 5e-7)
        close(kin["actual_grasp_position_error_m"], grasp, "Independent rigid grasp", 5e-7)
        for name, value in (("body_position_error_m", position), ("body_orientation_error_rad", orientation), ("grasp_error_m", grasp)):
            self.maximum[name] = max(self.maximum[name], value)
        if u >= g.prepare - 1e-12:
            mouth = pose[:3] + rotation @ np.array([0., 0., .18])
            self.maximum["dose_mouth_error_m"] = max(self.maximum["dose_mouth_error_m"], float(np.linalg.norm(mouth - g.data["pivot_m"])))
        self.previous = status, u, t
        self.rows += 1


def partitions(part, truth):
    keys = ("beaker", "tube", "airborne", "spilled", "below_table", "tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled")
    need(all(type(part[k]) is int and part[k] >= 0 for k in keys), "Invalid count")
    need(part["total"] == 11625 and part["initial_tube"] == 2880
         and sum(part[k] for k in keys[:4]) == 11625 and sum(part[k] for k in keys[5:]) == 2880,
         "Actual particle identity partitions changed")
    for amount, count in (("received_ml", "tube_to_beaker"), ("remaining_ml", "tube_remaining"),
                          ("inflight_ml", "tube_airborne"), ("spilled_ml", "tube_spilled")):
        close(truth[amount], part[count] * QUANTUM, "Mass/density volume: " + amount)


def assess(result, status, peak_spill, post):
    valid = bool(result["valid_physics"] and result["mass_conserved"] and result["all_states_finite"]
        and result["source_partition_conserved"] and result["max_below_table_particles"] == 0
        and abs(result["measurement_physical_time_s"] - result["physical_time_s"]) < 1e-8
        and post is not None and post["all_states_finite"] and post["mass_array_unchanged"]
        and post.get("maximum_below_table_particles", 0) == 0
        and abs(post.get("start_physical_time_s", -1) - result["physical_time_s"]) < 1e-8
        and post.get("physical_steps", 0) >= 120
        and post.get("end_physical_time_s", -1) - post.get("start_physical_time_s", 0) >= 2 - 1e-8)
    home = bool(result["complete"] and status.get("done") and status.get("home_static")
        and status.get("stable_frames", 0) >= 12 and status.get("source_airborne_particles") == 0
        and not status.get("timed_out") and result["physical_time_s"] <= 24)
    dose = abs(result["error_ml"]) <= .001 * result["target_ml"]
    no_late = post is not None and post["maximum_received_change_ml"] == 0.
    return {"valid_evidence": valid, "dose_passed": dose, "full_home_within_deadline": home,
            "maximum_all_label_spilled_particles": peak_spill, "spill_passed": peak_spill <= 1,
            "no_late_dose_change": no_late, "tolerance_ml": .001 * result["target_ml"],
            "feasible": bool(valid and dose and home and peak_spill <= 1 and no_late)}


def metrics(rows):
    errors = [abs(row["result"]["error_ml"]) for row in rows]
    return {"n": len(rows), "mae_ml": sum(errors) / len(rows), "worst_ml": max(errors),
            "feasible": sum(row["assessment"]["feasible"] for row in rows),
            "valid": sum(row["assessment"]["valid_evidence"] for row in rows),
            "dose_passed": sum(row["assessment"]["dose_passed"] for row in rows),
            "spill_particles": sum(row["assessment"]["maximum_all_label_spilled_particles"] for row in rows),
            "mean_physical_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
            "mean_wall_s": sum(row["wall_seconds"] for row in rows) / len(rows)}


def key(rows):
    m = metrics(rows)
    if m["valid"] != m["n"]:
        return 2, m["n"] - m["valid"], m["mae_ml"]
    if m["feasible"] == m["n"]:
        return 0, m["mean_physical_s"]
    return 1, m["mae_ml"], m["spill_particles"]


def audit_episode(run, row, plan, geometry):
    name, result = row["episode_id"], row["result"]
    need(read(inside(run, "completed/" + name + ".json")) == row, "Journal/completed sidecar differs")
    finite(row)
    role = row["role"]
    actual_controller = V9 + ("legacy_control.py" if role == "baseline" else "fine_control.py")
    action_name, post_name = f"traces/{name}.steps.jsonl.gz", f"traces/{name}.post.jsonl.gz"
    need(action_name in row["artifacts"] and set(row["artifacts"]) <= {action_name, post_name}, "Unexpected raw artifact set")
    for relative, digest in row["artifacts"].items():
        need(sha(inside(run, relative)) == digest, "Raw artifact SHA differs: " + relative)
    action, _ = gz(inside(run, action_name))
    header = action[0]
    need(header["type"] == "episode" and header["fluid_mode"] == "live_PBF_action_conditioned", "Live action header missing")
    same([header[k] for k in ("target_ml", "seed", "policy_name", "policy_params", "randomized")],
         [result[k] for k in ("target_ml", "seed", "policy_name", "policy_params", "randomized")], "Original header/footer conditions")
    need(result["seed"] == row["seed"] and result["target_ml"] == row["target_ml"] and result["randomized"] is True, "Scheduled conditions differ")
    close(header["physics_dt_s"], DT, "Action dt")
    close(header["particle_spacing_m"], .0027, "Unchanged physical particle spacing")
    close(header["particle_volume_ml"], QUANTUM, "True particle volume")
    close(header["density_kg_m3"], 1000., "Unchanged density")
    need(header["particle_count"] == result["particle_count"] == 11625 and result["source_particle_count"] == 2880, "Particle counts changed")
    close(result["initial_source_ml"], 56.68704, "Actual fixed source volume")
    close(result["particle_volume_ml"], QUANTUM, "Footer particle volume")
    close(result["physics_dt_s"], DT, "Footer physical dt")
    close(result["physical_time_s"], result["physical_steps"] * DT, "Original action physical time", 1e-8)
    need(result["physical_steps"] <= 1440, "Action exceeds declared raw24s")
    close(result["error_ml"], result["received_ml"] - result["target_ml"], "Signed error")
    need(header["minimum_substeps"] == 4 and result["minimum_substeps"] >= 4, "Adaptive minimum4 differs")
    effective = [x for x in action if x["type"] == "effective_direct_sources"]
    need(len(effective) == 1, "Unique actual binding required")
    effective = effective[0]
    expected_sources = {"mouth_pivot_lift_geometry.py": plan["source_sha256"][PREFIX + "mouth_pivot_lift_geometry.py"],
                        "run_fine_search.py": plan["source_sha256"][V9 + "run_fine_search.py"]}
    need(effective["source_sha256"] == expected_sources, "Actual geometry/factory source bytes")
    need(effective["actual_controller_file"] == actual_controller and effective["actual_controller_sha256"] == plan["source_sha256"][actual_controller], "Actual candidate/baseline implementation differs")
    need(effective["deadline_physical_seconds"] == 24 and effective["relative_volume_tolerance"] == .001
         and effective["maximum_all_label_spill_particles"] == 1 and effective["cuda_graph"] is False
         and effective["physics_solver_changed"] is False and effective["tube_containment"] == "actual_solver_body_q_full_R", "Effective task/physics differs")
    selected = "projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz"
    need(effective["cache_sha256"] == plan["input_sha256"][selected], "Effective cache differs")
    expected_kind = "timed_correction_closed_loop_v8" if role == "baseline" else "fine_particle_settled_micro_correction_v9"
    need(result["policy_name"] == expected_kind, "Role policy kind differs")
    params = header["policy_params"]
    close(params["dose_tilt_deg_s"], 25.6 * row["multiplier"], "Only coarse speed multiplier")
    for k, value in {"prepare_fraction": 1., "return_fraction": 1., "tail_lead_s": .26,
                     "stop_bias_ml": 10., "max_dose_u_accel": 6., "deadline_s": 24., "topup_tilt_deg_s": 2.2,
                     "topup_lead_s": .12, "reserve_ml": 0., "dose_duration_s": 0.,
                     "chunk_margin_ml": 1., "chunk_tilt_deg_s": 3.}.items():
        close(params[k], value, "Fixed parameter " + k)
    need(params["particle_trim"] is True, "Settled micro correction was disabled")
    need(params["enable_topup"] is True and all(params[name] is False for name in
         ("lip_pulses", "scale_feedback", "scale_pulses", "fine_percent")), "An unsearched correction mode changed")
    if role != "baseline":
        for k, value in {"micro_backoff_deg_s": 6., "settle_quiet_seconds": .2, "main_topup_headroom_ml": .15}.items():
            close(params[k], value, "Fixed candidate engineering value")
    steps = [x for x in action if x["type"] == "step"]
    direct = [x for x in action if x["type"] == "direct_step_evidence"]
    footers = [x for x in action if x["type"] == "result"]
    failures = [x for x in action if x["type"] == "failure"]
    need(len(footers) == 1 and action[-1] is footers[0], "One original final footer required")
    same({k: v for k, v in footers[0].items() if k != "type"}, result, "Original footer unchanged")
    need(len(steps) == len(direct) and bool(steps), "Every complete step needs actual evidence")
    motion, peak, below, drift = Motion(geometry), 0, 0, 0.
    initial_mass = result["actual_initial_total_mass_kg"]
    for index, (step, detail) in enumerate(zip(steps, direct)):
        need(step["step"] == detail["step"] == index and step["solver_substeps"] >= 4, "True step/substep sequence")
        close(step["physical_time_s"], index * DT, "True60Hz step time", 1e-8)
        close(detail["physical_time_s"], step["physical_time_s"], "Paired direct time", 1e-8)
        part = detail["partitions"]
        partitions(part, step["truth"])
        peak, below = max(peak, part["spilled"]), max(below, part["below_table"])
        close(detail["maximum_all_label_spilled_particles"], peak, "Cumulative all-label peak")
        drift = max(drift, abs(step["truth"]["mass_kg"] - initial_mass))
        motion.row(detail["physical_time_s"], step["source_time"], detail["kinematics"], detail["controller_status"])
    action_rows, action_dense = motion.rows, motion.dense_rows
    for failure in failures:
        need(not result["valid_physics"], "Failed action relabeled valid")
        motion.dense(failure.get("last_motion_evidence", {}).get("dense_240_hz_samples", []), terminal=True)
    gap = result["physical_steps"] - steps[-1]["step"]
    need(gap in (0, 1) and (gap == 0 or bool(failures)), "Unexplained action evidence gap")
    if gap == 0:
        close(result["received_ml"], steps[-1]["truth"]["received_ml"], "Footer actual dose")
    if result["complete"]:
        same(row["status"], direct[-1]["controller_status"], "Action-end status")
        need(len(direct) >= 13 and all(x["controller_status"]["home_static"] for x in direct[-13:]), "Complete original-home quiet interval missing")
    post, post_error = row["post_observation"], None
    post_valid = isinstance(post, dict) and post.get("all_states_finite") is True and post.get("mass_array_unchanged") is True and not post.get("error")
    post_rows, post_parse_error = (gz(inside(run, post_name), allow_partial=not post_valid)
                                  if post_name in row["artifacts"] else ([], None))
    maximum_change, last_time = 0., result["physical_time_s"]
    for index, item in enumerate(post_rows, 1):
        need(item["type"] == "post_observation" and item["step"] == index, "Post step sequence")
        close(item["physical_time_s"], result["physical_time_s"] + index * DT, "True2s post clock", 1e-8)
        part = item["measurement"]
        close(part["physical_time_s"], item["physical_time_s"], "Actual post measurement time", 1e-8)
        need(part["volume_transform_basis"] == "actual_solver_body_q_full_R", "Post actual-body full-R basis")
        partitions(part, part)
        peak, below = max(peak, part["spilled"]), max(below, part["below_table"])
        drift = max(drift, abs(part["total_mass_kg"] - initial_mass))
        maximum_change = max(maximum_change, abs(part["received_ml"] - result["received_ml"]))
        motion.row(item["physical_time_s"], part["source_time"], item["kinematics"], item["controller_status"])
        last_time = item["physical_time_s"]
    if post_valid:
        need(len(post_rows) == post["physical_steps"] == 120, "Exactly120 real post frames required")
        close(post["start_physical_time_s"], result["physical_time_s"], "Post actual start", 1e-8)
        close(post["end_physical_time_s"], last_time, "Post actual end", 1e-8)
        close(last_time - result["physical_time_s"], 2., "Two original physical seconds", 1e-8)
        close(post["maximum_received_change_ml"], maximum_change, "Actual post dose change")
        need(post["maximum_below_table_particles"] == below, "Post below-table peak differs")
        need(row["assessment"]["maximum_all_label_spilled_particles"] == peak, "All-label peak spill differs")
    else:
        need(not row["assessment"]["valid_evidence"], "Partial/invalid post rescued a case")
        post_error = post.get("error") if isinstance(post, dict) else "not_run_after_invalid_action"
    expected = assess(result, row["status"], row["assessment"]["maximum_all_label_spilled_particles"], post)
    same(row["assessment"], expected, "Independent task assessment (not old footer.success)")
    if expected["valid_evidence"]:
        need(drift <= 1e-12 and result["particle_mass_array_unchanged"] and result["mass_volume_units_consistent"], "Valid mass evidence contradicts raw sums")
        limits = {"speed_limit_ratio": 1.00001, "acceleration_rad_s2": 25.0001,
                  "position_violation_rad": 1e-6, "body_position_error_m": 1e-5,
                  "body_orientation_error_rad": 1e-4, "grasp_error_m": .003}
        need(all(motion.maximum[k] <= bound for k, bound in limits.items()), "Independent motion contradicts valid case")
        close(result["actual_initial_source_mass_kg"] * 1e6 / header["density_kg_m3"],
              56.68704, "Actual source mass/density volume", 2e-4)
    return {"episode_id": name, "stage": row["stage"], "role": role, "seed": row["seed"],
            "target_ml": row["target_ml"], "action_rows": action_rows, "action_dense_rows": action_dense,
            "all_dense_rows": motion.dense_rows, "all_FK_rows": motion.rows,
            "maximum_motion": motion.maximum, "maximum_logged_mass_drift_kg": drift,
            "maximum_logged_all_label_spill_particles": peak, "maximum_below_table_particles": below,
            "post_rows": len(post_rows), "post_change_ml": maximum_change,
            "post_error": post_error, "post_partial_parse_error": post_parse_error,
            "unlogged_terminal_action_frames": gap, "failures": failures,
            "assessment": expected, "actual_reset_started_utc": effective["reset_started_utc"],
            "old_footer_success_used_for_selection": False}


def expected_ids():
    ids = [f"train_candidate_g{g}c{c}_t{t}_s171101" for g in range(2) for c in range(3) for t in TARGETS]
    ids += [f"validation_{role}__t{t}_s172101" for role in ("candidate", "baseline") for t in TARGETS]
    ids += [f"test_{role}__t{t}_s{seed}" for role in ("candidate", "baseline") for seed in TEST for t in TARGETS]
    return ids


def audit(run, root):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary = completion_gate(run)  # Always first experiment read.
    plan = read(run / "protocol.json")
    need(plan["kind"] == "fine_particle_24s_coarse_speed_CEM_v9", "Different experiment kind")
    need(signature({k: v for k, v in plan.items() if k != "protocol_id"}) == plan["protocol_id"], "Canonical protocol identity")
    if "protocol_id" in summary:
        need(summary["protocol_id"] == plan["protocol_id"], "Summary protocol identity")
    need(plan["budget"] == BUDGET and tuple(plan["targets_ml"]) == TARGETS, "Frozen budget/target set")
    same(plan["search"], {"generations": 2, "population": 3, "elites": 2, "seed": 2026112709,
                          "parameter": "coarse_tilt_speed_multiplier", "bounds": [.92, 1.02],
                          "center": 1., "initial_std": .025, "actual_deg_s": [25.6*.92, 25.6*1.02]},
         "Only the frozen coarse-speed search may be replayed")
    need(tuple(plan["train_seeds"]) == TRAIN and tuple(plan["validation_seeds"]) == VAL and tuple(plan["test_seeds"]) == TEST, "Frozen seed split")
    need(plan["user_criterion"] == {"deadline_s": 24., "relative_error": .001, "max_all_label_spill_particles": 1,
                                  "full_original_home_and_12_quiet_frames": True}, "Strict user criterion changed")
    need(len(plan["source_sha256"]) == 19 and len(plan["input_sha256"]) == 6, "Expected19source/6input freeze")
    need(plan["source_sha256"][V9 + "run_fine_search.py"] == RUNNER_SHA256, "Runner is not the reviewed frozen implementation")
    for group, snapshot in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for relative, digest in plan[group].items():
            need(sha(inside(root, relative)) == sha(inside(run / snapshot, relative)) == digest, "Native/frozen bytes differ: " + relative)
    close(plan["particle_ml"], QUANTUM, "Actual particle quantum")
    need(plan["initial_particle_count"] == 11625 and plan["initial_source_particles"] == 2880, "Fixed fine-particle counts")
    close(plan["initial_source_ml"], 56.68704, "Fixed source56.68704mL")
    if "initial_source_ml" in summary:
        close(summary["initial_source_ml"], 56.68704, "Summary source is actual fine-particle fill")
    need(len(plan["gui_sha256"]) == 3 and all(sha(inside(root, path)) == digest for path, digest in plan["gui_sha256"].items()), "Old3 GUI models changed")
    records, order = {}, []
    for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line, parse_constant=reject_constant)
        name = row["episode_id"]
        need(name not in records, "Duplicate journal identity")
        if "protocol_id" in row:
            need(row["protocol_id"] == plan["protocol_id"], "Foreign journal protocol")
        records[name], order = row, order + [name]
    need(order == expected_ids() and len(records) == 42, "Exactly18/6/18 scheduled cases, all retained")
    need(Counter(row["stage"] for row in records.values()) == {k: BUDGET[k] for k in ("train", "validation", "test")}, "Actual stage budget")
    geometry = Geometry(run, root, plan)
    reports = [audit_episode(run, records[name], plan, geometry) for name in order]
    rng, mean, std, best = random.Random(2026112709), 1., .025, None
    generations = []
    for generation in range(2):
        proposals = [mean]
        if best and best["multiplier"] != mean:
            proposals.append(best["multiplier"])
        while len(proposals) < 3:
            proposals.append(min(1.02, max(.92, rng.gauss(mean, std))))
        candidates = []
        for index, multiplier in enumerate(proposals):
            group = [records[f"train_candidate_g{generation}c{index}_t{t}_s171101"] for t in TARGETS]
            for row in group:
                same([row[k] for k in ("stage", "role", "multiplier")], ["train", "candidate", multiplier], "Actual CEM conditions")
            candidate = {"generation": generation, "index": index, "multiplier": multiplier,
                         "key": key(group), "metrics": metrics(group)}
            candidates.append(candidate)
            if candidate["key"][0] < 2 and (best is None or candidate["key"] < best["key"]):
                best = candidate
        eligible = sorted((c for c in candidates if c["key"][0] < 2), key=lambda c: c["key"])[:2]
        need(bool(eligible), "No physically valid candidate can enter elites")
        average = sum(c["multiplier"] for c in eligible) / len(eligible)
        deviation = math.sqrt(sum((c["multiplier"] - average) ** 2 for c in eligible) / len(eligible))
        mean, std = .65 * mean + .35 * average, max(.005, .65 * std + .35 * deviation)
        entry = {"generation": generation, "candidates": candidates, "next_mean": mean, "next_std": std}
        same(summary["generations"][generation], entry, "Independent CEM replay")
        same(read(run / f"generation_{generation}.json"), entry, "Generation snapshot")
        generations.append(entry)
    validation = {role: [records[f"validation_{role}__t{t}_s172101"] for t in TARGETS] for role in ("candidate", "baseline")}
    for role, group in validation.items():
        need(all(row["multiplier"] == (best["multiplier"] if role == "candidate" else 1.) for row in group), "Actual validation policy differs")
        same(summary["validation"][role], metrics(group), "Validation metrics")
    recommendation = "candidate" if key(validation["candidate"]) <= key(validation["baseline"]) else "baseline"
    need(summary["recommended_role"] == recommendation, "Test changed validation-only recommendation")
    freeze = read(run / "freeze.json")
    same(summary["freeze"], freeze, "Final/pre-test freeze bytes")
    need(freeze["protocol_id"] == plan["protocol_id"] and freeze["recommendation"] == recommendation
         and freeze["source_sha256"] == plan["source_sha256"] and freeze["validation_only_selection"] is True
         and freeze["test_used_for_selection"] is False and freeze["gui_deployment"] is False, "Pre-test freeze provenance")
    frozen_at = utc(freeze["frozen_at_utc"])
    need(max(utc(row["completed_at_utc"]) for group in validation.values() for row in group) < frozen_at, "Validation was not completed before freeze")
    model_hashes = {}
    need(set(freeze["models_sha256"]) == {"candidate_frozen.json", "recommended_frozen.json"}, "Exactly two pre-test frozen model files")
    for role, implementation in (("candidate", "candidate"), ("recommended", recommendation)):
        name = role + "_frozen.json"
        model = read(run / name)
        model_hashes[name] = sha(run / name)
        need(model_hashes[name] == freeze["models_sha256"][name], "Frozen raw model SHA")
        need(model["role"] == role and model["implementation"] == implementation and model["source_sha256"] == plan["source_sha256"]
             and model["protocol_id"] == plan["protocol_id"] and model["test_used_for_selection"] is False and model["gui_deployment"] is False, "Model role/provenance")
        close(model["coarse_speed_multiplier"], best["multiplier"] if implementation == "candidate" else 1., "Frozen policy multiplier")
        need(model["policy_kind"] == ("fine_particle_settled_micro_correction_v9" if implementation == "candidate" else
                                      "timed_correction_closed_loop_v8") and model["particle_trim"] is True,
             "Frozen implementation kind or correction mode differs")
        same(model["coarse_base_parameters"], {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
             "return_fraction": 1., "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.},
             "Frozen unsearched coarse parameters")
        for field, expected in {"deadline_s": 24., "spacing_m": .0027, "topup_tilt_deg_s": 2.2,
                                "topup_lead_s": .12, "fixed_main_topup_margin_ml": .15 if implementation == "candidate" else 0.,
                                "fixed_micro_backoff_deg_s": 6. if implementation == "candidate" else 2.}.items():
            close(model[field], expected, "Frozen model fixed value " + field)
    tests = {role: [records[f"test_{role}__t{t}_s{seed}"] for seed in TEST for t in TARGETS] for role in ("candidate", "baseline")}
    by_id = {report["episode_id"]: report for report in reports}
    first_test = min(utc(by_id[row["episode_id"]]["actual_reset_started_utc"]) for group in tests.values() for row in group)
    need(first_test > frozen_at, "An actual test reset preceded frozen models")
    for role, group in tests.items():
        need(all(row["multiplier"] == (best["multiplier"] if role == "candidate" else 1.) for row in group), "Actual test policy differs")
        same(summary["test"][role], metrics(group), "Independent heldout metrics")
    task_passed = all(row["assessment"]["feasible"] for row in tests[recommendation])
    need(summary["task_passed"] is task_passed and summary["gui_deployment"] is False, "Task completion confused with successful experiment")
    same(read(run / "versions.json"), summary["versions"], "Software versions")
    all_artifacts = {name for row in records.values() for name in row["artifacts"]}
    need(not [p.name for p in (run / "traces").glob("*.steps.jsonl.gz") if p.relative_to(run).as_posix() not in all_artifacts], "Unjournaled physical attempt must not be hidden")
    return {"audit_schema_version": 1, "kind": "independent_V9_fine42_live_audit",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "audit_tool_sha256": sha(__file__),
            "protocol_id": plan["protocol_id"], "summary_sha256": sha(run / "summary.json"),
            "journal_sha256": sha(run / "episodes.jsonl"), "budgets": BUDGET,
            "artifact_integrity_verified": True, "all_failures_retained": True, "episodes_audited": 42,
            "raw_action_trace_count": 42, "source_files_verified": 19, "input_files_verified": 6,
            "source_count": 19, "input_count": 6,
            "raw_URDF_reference_sha256": URDF_SHA256, "old_3_GUI_models_unchanged": True,
            "models_raw_sha256": model_hashes, "validation_only_recommendation_verified": True,
            "CEM_replay_verified": True, "freeze_to_first_test_reset_seconds": (first_test - frozen_at).total_seconds(),
            "initial_source_particles": 2880, "initial_source_ml": 56.68704, "particle_ml": QUANTUM,
            "joint_240hz_samples_audited": sum(x["all_dense_rows"] for x in reports),
            "FK_full_R_60hz_rows_audited": sum(x["all_FK_rows"] for x in reports),
            "test_metrics": {role: metrics(group) for role, group in tests.items()},
            "recommended_role": recommendation, "task_passed": task_passed,
            "episode_audits": by_id, "failed_task_case_ids": [r["episode_id"] for r in records.values() if not r["assessment"]["feasible"]],
            "invalid_evidence_case_ids": [r["episode_id"] for r in records.values() if not r["assessment"]["valid_evidence"]],
            "no_GPU_or_liquid_execution": True,
            "limitations": ["Aggregate particle counts and mass sums are independently checked; full per-particle state/mass arrays are not stored in these traces.",
                "Particle mass/density is privileged simulator metrology, not a calibrated physical scale or image sensor.",
                "240Hz joint checks and60Hz actual-body FK are sampled; no torque, frictional grasp or continuous collision certificate is claimed.",
                "The raw SDK URDF is an explicit pinned independent audit reference, beyond the six frozen experiment inputs.",
                "Failed/unlogged terminal frames stay invalid; available prefixes cannot establish unseen states or rescue a24s miss.",
                "Only three fixed training conditions, one validation seed and three independent test seeds; CEM completion is not task success."]}


def self_test():
    """Meaningful synthetic counterexamples; no real journal/model/test read."""
    checks = []
    with tempfile.TemporaryDirectory(prefix="fine_audit_cpu_", dir=Path(__file__).parent) as name:
        root = Path(name)
        need(root.resolve().is_relative_to(Path(__file__).resolve().parent), "Unsafe synthetic temporary directory")
        (root / "protocol.json").write_text("MUST NOT PARSE PARTIAL PROTOCOL", encoding="utf-8")
        (root / "episodes.jsonl").write_text("MUST NOT READ PARTIAL JOURNAL OR TEST", encoding="utf-8")
        for data in (None, {"complete": False, "counts": {"train": 18, "validation": 6, "test": 18}},
                     {"complete": True, "counts": {"train": 17, "validation": 6, "test": 18}}):
            if data is not None:
                (root / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            try:
                audit(root, ROOT)
            except AuditError:
                checks.append("partial42_gate_refused_before_journal")
            else:
                raise AuditError("Partial experiment passed")
    part = {"total": 11625, "initial_tube": 2880, "beaker": 8745, "tube": 2880,
            "airborne": 0, "spilled": 0, "below_table": 0, "tube_to_beaker": 0,
            "tube_remaining": 2880, "tube_airborne": 0, "tube_spilled": 0}
    truth = {"received_ml": 0., "remaining_ml": 2880 * QUANTUM, "inflight_ml": 0., "spilled_ml": 0.}
    partitions(part, truth)
    try:
        partitions({**part, "tube_remaining": 2879}, truth)
    except AuditError:
        checks.append("source_identity_loss_rejected")
    else:
        raise AuditError("Wrong source identity count passed")
    result = {"valid_physics": True, "mass_conserved": True, "all_states_finite": True,
              "source_partition_conserved": True, "max_below_table_particles": 0,
              "measurement_physical_time_s": 20., "physical_time_s": 20., "complete": True,
              "target_ml": 20., "error_ml": .02}
    status = {"done": True, "home_static": True, "stable_frames": 13,
              "source_airborne_particles": 0, "timed_out": False}
    post = {"all_states_finite": True, "mass_array_unchanged": True,
            "maximum_below_table_particles": 0, "start_physical_time_s": 20.,
            "end_physical_time_s": 22., "physical_steps": 120, "maximum_received_change_ml": 0.}
    need(assess(result, status, 1, post)["feasible"], "Inclusive0.1percent/onegrain boundary")
    need(not assess({**result, "error_ml": math.nextafter(.02, math.inf)}, status, 1, post)["feasible"], "Strict dose threshold")
    need(not assess(result, status, 2, post)["feasible"], "All-label spill must not disappear")
    need(not assess(result, status, 1, {**post, "maximum_received_change_ml": QUANTUM})["feasible"], "Post observation cannot rescue dose")
    checks += ["strict_percentage_boundary", "two_grain_rejected", "post_new_water_rejected"]
    def synthetic(error, spill=0, valid=True, feasible=False, elapsed=10.):
        return {"result": {"error_ml": error, "physical_time_s": elapsed}, "wall_seconds": 1.,
                "assessment": {"valid_evidence": valid, "feasible": feasible,
                               "dose_passed": abs(error) <= .02,
                               "maximum_all_label_spilled_particles": spill}}
    need(key([synthetic(.01, 2)]) < key([synthetic(-10., 0)]), "Volume error must precede spill for valid failures")
    need(key([synthetic(.01, feasible=True, elapsed=12.)]) < key([synthetic(0., feasible=True, elapsed=14.)]), "Feasible ranks by raw action time")
    need(key([synthetic(.1, elapsed=8.)]) == key([synthetic(.1, elapsed=20.)]), "Infeasible time must not break ties")
    need(key([synthetic(0., valid=False)])[0] == 2, "Invalid evidence cannot enter the eligible tiers")
    checks += ["volume_precedes_spill", "only_feasible_compares_time", "invalid_evidence_tier_excluded"]
    return {"self_test_passed": True, "checks": checks, "real_journal_or_test_read": False, "GPU_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    report = audit(args.run, args.root)
    output = args.output or args.run / "fine_run_audit.json"
    if output.exists():
        previous = read(output)
        need(previous.get("kind") == report["kind"] and previous.get("protocol_id") == report["protocol_id"], "Refuse to replace unrelated output")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_file": str(output), "sha256": sha(output),
                      "episodes_audited": 42, "artifact_integrity_verified": True,
                      "task_passed": report["task_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print("Audit refused: " + str(error), file=sys.stderr)
        raise SystemExit(2)
