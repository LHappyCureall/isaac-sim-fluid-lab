"""Independent CPU audit of a completed V3 138-episode fast-pouring run.

Partial runs are refused before opening the journal, models or any trial trace.
No controller, collector, Warp, Kit or training module is imported. NumPy is
used only for frozen path-polynomial/URDF FK arithmetic after the complete gate.
Original action results and failed trials are never rewritten or filtered.
"""

import argparse
import ast
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab_paths import require_isaac_sim_root
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_fast_v3/run_20261122_cem5s"
PREFIX = "assets/robot_testtube_pour/scripts/"
DT, QUANTUM, TOTAL, SOURCE = 1 / 60, .0035 ** 3 * 1e6, 5330, 1198
MASS_ONE = struct.unpack("f", struct.pack("f", 1000 * .0035 ** 3))[0]
MASS_TOTAL, MASS_SOURCE = TOTAL * MASS_ONE, SOURCE * MASS_ONE


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(AuditError("Nonfinite JSON token " + value)))


def finite(value):
    if isinstance(value, float):
        need(math.isfinite(value), "Nonfinite logged number")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def close(one, two, label, tol=1e-9):
    need(type(one) in (int, float) and math.isfinite(one) and abs(one - two) <= tol,
         f"{label}: {one!r} != {two!r} (tolerance {tol:g})")


def equal(one, two, label="value"):
    if isinstance(two, dict):
        need(isinstance(one, dict) and set(one) == set(two), label + ": keys differ")
        for key in two:
            equal(one[key], two[key], label + "." + key)
    elif isinstance(two, list):
        need(isinstance(one, list) and len(one) == len(two), label + ": length differs")
        for index, (a, b) in enumerate(zip(one, two)):
            equal(a, b, f"{label}[{index}]")
    elif type(two) is float:
        close(one, two, label, max(1e-9, abs(two) * 2e-12))
    else:
        need(one == two, label + ": value differs")


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "Timestamp lacks explicit UTC")
    return result


def inside(root, relative):
    result = (Path(root) / relative).resolve()
    need(result.is_relative_to(Path(root).resolve()), "Artifact path escapes its root")
    return result


def paths(mapping):
    result = {key.replace("\\", "/"): value for key, value in mapping.items()}
    need(len(result) == len(mapping), "Duplicate normalized manifest paths")
    return result


def numpy_cpu(root):
    try:
        import numpy as np
    except ImportError:
        sdk = require_isaac_sim_root()
        for candidate in sorted((sdk / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")):
            sys.path.insert(0, str(candidate))
        import numpy as np
    return np


def source_constants(path):
    result = {}
    def literal(node):
        if isinstance(node, ast.Name):
            return result[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(literal(item) for item in node.elts)
        return ast.literal_eval(node)
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        result[target.id] = literal(node.value)
                    except (ValueError, KeyError, TypeError):
                        pass
    return result


class FrozenPath:
    """Independent coefficient evaluation and FK from the raw frozen URDF."""

    def __init__(self, run, protocol, root):
        self.np = np = numpy_cpu(root)
        cache = run / "input_snapshot/path_cache.npz"
        with np.load(cache, allow_pickle=False) as archive:
            self.cache = {key: archive[key].copy() for key in archive.files}
        self.u_grid = self.cache["u"]
        self.source_grid = self.cache["joint_spline_source"]
        self.maximum_u = float(self.u_grid[-1])
        self.c = source_constants(run / "source_snapshot" / PREFIX / "geometry.py")
        self.knots = [(2 * float(t), np.asarray(point, dtype=float), math.radians(angle))
                      for t, point, angle in self.c["MOTION_KNOTS"]]
        self.lower = np.asarray(protocol["robot_envelope"]["joint_lower_limits_rad"])
        self.upper = np.asarray(protocol["robot_envelope"]["joint_upper_limits_rad"])
        self.vlimits = np.asarray(protocol["robot_envelope"]["joint_velocity_limits_rad_s"])
        need(all(value.shape == (7,) for value in (self.lower, self.upper, self.vlimits)), "Seven joint limits required")
        urdf = ET.parse(run / "input_snapshot/urdf.urdf").getroot()
        joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
        self.names = [str(name) for name in self.cache["joint_names"]]
        self.origins = []
        for index, name in enumerate(self.names):
            joint = joints[name]
            axis = joint.find("axis")
            need(axis is not None and list(map(float, axis.get("xyz").split())) == [0, 0, 1], "Unexpected URDF rotation axis")
            limit = joint.find("limit")
            for key, array in (("lower", self.lower), ("upper", self.upper), ("velocity", self.vlimits)):
                close(float(limit.get(key)), float(array[index]), "Raw URDF joint " + key)
            self.origins.append(self.origin(joint))
        child_joints = {joint.find("child").get("link"): joint for joint in joints.values()}
        current = "panda_hand"
        end = joints[self.names[-1]].find("child").get("link")
        fixed = []
        while current != end:
            joint = child_joints[current]
            need(joint.get("type") == "fixed", "Unexpected movable joint in hand fixed chain")
            fixed.append(self.origin(joint))
            current = joint.find("parent").get("link")
        self.hand_fixed = np.eye(4)
        for matrix in reversed(fixed):
            self.hand_fixed = self.hand_fixed @ matrix
        self.base = self.cache["base_world_matrix_col"]
        self.vector_close(self.cache["origin_matrices_col"], self.origins, "Cached URDF joint origins", 3e-7)
        self.vector_close(self.cache["hand_fixed_matrix_col"], self.hand_fixed, "Cached URDF hand fixed chain", 3e-7)
        close(self.maximum_u, protocol["path_coordinate"]["u_max"], "u domain")
        for key, value in protocol["cache_input_sha256"].items():
            need(str(self.cache[key]) == value, "Embedded raw-cache input digest differs: " + key)

    def vector_close(self, one, two, label, tol=1e-8):
        a, b = self.np.asarray(one, dtype=float), self.np.asarray(two, dtype=float)
        need(a.shape == b.shape and self.np.isfinite(a).all() and self.np.max(self.np.abs(a - b)) <= tol,
             label + ": shape/nonfinite/numeric discrepancy")

    def origin(self, joint):
        np = self.np
        origin = joint.find("origin")
        xyz = [0., 0., 0.] if origin is None else list(map(float, origin.get("xyz", "0 0 0").split()))
        r, p, y = [0., 0., 0.] if origin is None else list(map(float, origin.get("rpy", "0 0 0").split()))
        rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        ry = self.ry(p)
        rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = rz @ ry @ rx, xyz
        return result

    def ry(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return self.np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def polynomial(self, grid, coefficients, value, derivative=0):
        np = self.np
        index = min(max(int(np.searchsorted(grid, value, side="right")) - 1, 0), len(grid) - 2)
        x = (value - grid[index]) / (grid[index + 1] - grid[index])
        result = np.zeros_like(coefficients[index, 0])
        h = grid[index + 1] - grid[index]
        for power in range(derivative, len(coefficients[index])):
            factor = math.factorial(power) / math.factorial(power - derivative)
            result += coefficients[index, power] * factor * x ** (power - derivative) / h ** derivative
        return result

    def source(self, u):
        need(-1e-8 <= u <= self.maximum_u + 1e-8, "Actual commanded u leaves frozen path")
        value = self.polynomial(self.u_grid, self.cache["source_coefficients"], min(self.maximum_u, max(0., u)))
        return float(self.np.clip(value[0], 0, float(self.cache["source"][-1])))

    def q(self, u):
        return self.polynomial(self.source_grid, self.cache["joint_coefficients"], self.source(u))

    def q_u(self, u):
        source = self.source(u)
        derivative = self.polynomial(self.u_grid, self.cache["source_coefficients"], min(self.maximum_u, max(0., u)), 1)[0]
        return self.polynomial(self.source_grid, self.cache["joint_coefficients"], source, 1) * derivative

    def poses(self, u, q):
        np = self.np
        source = self.source(u)
        for first, second in zip(self.knots[:-1], self.knots[1:]):
            if source <= second[0] + 1e-12:
                x = float(np.clip((source - first[0]) / (second[0] - first[0]), 0, 1))
                alpha = x ** 3 * (10 - 15 * x + 6 * x * x)
                mouth = first[1] + alpha * (second[1] - first[1])
                angle = first[2] + alpha * (second[2] - first[2])
                break
        else:
            raise AuditError("Source outside analytic pose knots")
        tcp = mouth - self.ry(angle) @ np.array([0., 0., self.c["TCP_BELOW_MOUTH"]])
        hand_rotation = self.ry(angle + .5 * math.pi)
        hand = tcp - hand_rotation @ np.array([0., 0., self.c["PANDA_HAND_TO_TCP"]])
        matrix = self.base.copy()
        for origin, value in zip(self.origins, q):
            rotation = np.eye(4)
            c, s = math.cos(float(value)), math.sin(float(value))
            rotation[:2, :2] = [[c, -s], [s, c]]
            matrix = matrix @ origin @ rotation
        matrix = matrix @ self.hand_fixed
        position_error = float(np.linalg.norm(matrix[:3, 3] - hand))
        rotation_error = float(np.arccos(np.clip((np.trace(matrix[:3, :3].T @ hand_rotation) - 1) / 2, -1, 1)))
        return tcp, math.degrees(angle), position_error, rotation_error


def commanded_u(status, time_s):
    elapsed = max(0., time_s - status["command_start_physical_s"])
    horizon = status["command_horizon_s"]
    h = min(elapsed, horizon)
    v = status["command_source_velocity"]
    a = status["command_acceleration"]
    value = status["command_start_u"] + v * h + .5 * a * h * h
    velocity = v + a * h
    # A shortened within-frame horizon occurs only at a stop followed by rest.
    if elapsed > horizon + 1e-10:
        need(abs(velocity) <= 1e-7, "Unlogged non-rest continuation within the recorded frame")
    return value + velocity * max(0., elapsed - horizon), velocity


class KinematicAudit:
    def __init__(self, path):
        self.path, self.np = path, path.np
        self.previous_time = self.previous_q = self.previous_tcp = self.previous_tilt = None
        self.previous_qd = self.np.zeros(7)
        self.dense_time = self.dense_q = None
        self.dense_qd = self.np.zeros(7)
        self.previous_status = None
        self.maxima = {key: 0. for key in ("maximum_joint_speed_limit_ratio", "maximum_joint_acceleration_rad_s2",
            "maximum_tcp_speed_m_s", "maximum_tilt_speed_deg_s", "maximum_hand_fk_position_error_m",
            "maximum_hand_fk_orientation_error_rad")}
        self.rows, self.dense_rows, self.max_position_violation = 0, 0, 0.
        self.command_residual = 0.
        self.home_checks = deque(maxlen=13)
        self.inferred_terminal_statuses = 0

    def terminal_without_status(self, kinematics):
        """Audit retained failed post-frame q evidence without inventing water truth."""
        dense = kinematics.get("dense_240_hz_samples", [])
        if not dense or self.previous_time is None or dense[-1]["time_s"] <= self.previous_time + 1e-10:
            return
        time_s = dense[-1]["time_s"]
        u, velocity = commanded_u(self.previous_status, time_s)
        q = self.np.asarray(kinematics["joint_position_rad"])
        tcp, tilt, _, _ = self.path.poses(u, q)
        origin = tcp - self.path.ry(math.radians(tilt)) @ self.np.array([0., 0., self.path.c["TUBE_LENGTH"] - self.path.c["TCP_BELOW_MOUTH"]])
        home_tcp, _, _, _ = self.path.poses(0., self.path.q(0.))
        home_origin = home_tcp - self.np.array([0., 0., self.path.c["TUBE_LENGTH"] - self.path.c["TCP_BELOW_MOUTH"]])
        pe = float(self.np.linalg.norm(origin - home_origin))
        angle = abs(math.radians(tilt))
        je = float(self.np.abs(q - self.path.q(0.)).max())
        joint_velocity = self.path.q_u(u) * velocity
        status = {**self.previous_status, "u": u, "source_time": u, "actual_u_velocity": velocity,
                  "home_position_error_m": pe, "home_orientation_error_rad": angle, "home_joint_error_rad": je,
                  "joint_velocity_rad_s": joint_velocity.tolist(),
                  "home_static": bool(pe <= .0005 and angle <= math.radians(.1) and je <= .001
                    and abs(velocity) <= 1e-4 and float(self.np.abs(joint_velocity).max()) <= .001)}
        self.observe(time_s, kinematics, status, terminal=True)
        self.inferred_terminal_statuses += 1

    def observe(self, time_s, kinematics, status, *, terminal=False):
        np, p = self.np, self.path
        q = np.asarray(kinematics["joint_position_rad"], dtype=float)
        need(q.shape == (7,) and np.isfinite(q).all(), "Invalid observed seven-joint state")
        if self.previous_time is not None and time_s <= self.previous_time + 1e-10:
            need(abs(time_s - self.previous_time) <= 1e-10, "Kinematic time runs backward")
            p.vector_close(q, self.previous_q, "Duplicate terminal q")
            return
        u = status.get("u", status.get("source_time"))
        p.vector_close(q, p.q(u), "Observed q vs independently evaluated frozen spline")
        self.max_position_violation = max(self.max_position_violation,
            float(np.maximum(np.maximum(p.lower - q, q - p.upper), 0).max()))
        p.vector_close(kinematics["joint_velocity_limits_rad_s"], p.vlimits, "Logged vs raw URDF velocity limits")
        tcp, tilt, pe, re = p.poses(u, q)
        origin = tcp - p.ry(math.radians(tilt)) @ np.array([0., 0., p.c["TUBE_LENGTH"] - p.c["TCP_BELOW_MOUTH"]])
        home_tcp, _, _, _ = p.poses(0., p.q(0.))
        home_origin = home_tcp - np.array([0., 0., p.c["TUBE_LENGTH"] - p.c["TCP_BELOW_MOUTH"]])
        home_position = float(np.linalg.norm(origin - home_origin))
        home_angle = abs(math.radians(tilt))
        home_joint = float(np.abs(q - p.q(0.)).max())
        close(status["home_position_error_m"], home_position, "Independent home tube-origin error", 1e-8)
        close(status["home_orientation_error_rad"], home_angle, "Independent home orientation error", 2e-7)
        close(status["home_joint_error_rad"], home_joint, "Independent home joint error", 1e-8)
        analytic_joint_velocity = p.q_u(u) * status["actual_u_velocity"]
        p.vector_close(status["joint_velocity_rad_s"], analytic_joint_velocity, "Controller joint velocity vs frozen derivative", 1e-7)
        home_static = bool(home_position <= .0005 and home_angle <= math.radians(.1) and home_joint <= .001
                           and abs(status["actual_u_velocity"]) <= 1e-4 and float(np.abs(analytic_joint_velocity).max()) <= .001)
        need(status["home_static"] == home_static, "Home-static flag does not match actual path/velocity")
        self.home_checks.append({"home_static": home_static, "received_ml": status["received_ml"],
                                 "airborne": status["source_airborne_particles"]})
        p.vector_close(kinematics["tcp_position_m"], tcp, "60Hz TCP position")
        close(kinematics["hand_fk_position_error_m"], pe, "Independent raw-URDF FK position error", 1e-8)
        close(kinematics["hand_fk_orientation_error_rad"], re, "Independent raw-URDF FK angle error", 6e-7)
        self.maxima["maximum_hand_fk_position_error_m"] = max(self.maxima["maximum_hand_fk_position_error_m"], pe)
        self.maxima["maximum_hand_fk_orientation_error_rad"] = max(self.maxima["maximum_hand_fk_orientation_error_rad"], re)
        if self.previous_time is None:
            self.dense_q, self.dense_time = q.copy(), time_s
            qd = qdd = np.zeros(7)
            need(not kinematics["dense_240_hz_samples"], "Initial state unexpectedly contains elapsed dense samples")
        else:
            dt = time_s - self.previous_time
            close(dt, DT, "60Hz endpoint interval", 1e-10)
            qd = (q - self.previous_q) / dt
            qdd = (qd - self.previous_qd) / dt
            self.maxima["maximum_tcp_speed_m_s"] = max(self.maxima["maximum_tcp_speed_m_s"], float(np.linalg.norm(tcp - self.previous_tcp) / dt))
            self.maxima["maximum_tilt_speed_deg_s"] = max(self.maxima["maximum_tilt_speed_deg_s"], abs(tilt - self.previous_tilt) / dt)
            samples = kinematics["dense_240_hz_samples"]
            need(len(samples) == 4, "A complete kinematic frame lacks four 240Hz joint samples")
            for sample in samples:
                t = sample["time_s"]
                h = t - self.dense_time
                close(h, DT / 4, "240Hz joint interval", 1e-10)
                dense_q = np.asarray(sample["q_rad"])
                dense_v = (dense_q - self.dense_q) / h
                dense_a = (dense_v - self.dense_qd) / h
                p.vector_close(sample["dq_rad_s"], dense_v, "240Hz actual dq finite difference", 1e-7)
                p.vector_close(sample["ddq_rad_s2"], dense_a, "240Hz actual ddq finite difference", 2e-5)
                if self.previous_status is not None:
                    query_u, _ = commanded_u(self.previous_status, t)
                    p.vector_close(dense_q, p.q(query_u), "Dense q vs committed historical command", 1e-7)
                ratio = float(np.max(np.abs(dense_v) / p.vlimits))
                close(sample["speed_limit_ratio"], ratio, "240Hz joint speed ratio", 1e-7)
                self.maxima["maximum_joint_speed_limit_ratio"] = max(self.maxima["maximum_joint_speed_limit_ratio"], ratio)
                self.maxima["maximum_joint_acceleration_rad_s2"] = max(self.maxima["maximum_joint_acceleration_rad_s2"], float(np.abs(dense_a).max()))
                self.dense_q, self.dense_qd, self.dense_time = dense_q.copy(), dense_v, t
                self.dense_rows += 1
            p.vector_close(samples[-1]["q_rad"], q, "Dense/60Hz endpoint q")
            if self.previous_status is not None:
                predicted, velocity = commanded_u(self.previous_status, time_s)
                self.command_residual = max(self.command_residual, abs(predicted - u))
                close(u, predicted, "Continuous within-frame command integration", 1e-7)
                close(status["actual_u_velocity"], velocity, "C1 boundary velocity", 1e-6)
        p.vector_close(kinematics["joint_velocity_rad_s"], qd, "60Hz dq finite difference", 1e-7)
        p.vector_close(kinematics["joint_acceleration_rad_s2"], qdd, "60Hz ddq finite difference", 2e-5)
        ratio = float(np.max(np.abs(qd) / p.vlimits))
        close(kinematics["joint_speed_limit_ratio"], ratio, "60Hz speed ratio", 1e-7)
        self.maxima["maximum_joint_speed_limit_ratio"] = max(self.maxima["maximum_joint_speed_limit_ratio"], ratio)
        self.maxima["maximum_joint_acceleration_rad_s2"] = max(self.maxima["maximum_joint_acceleration_rad_s2"], float(np.abs(qdd).max()))
        self.previous_time, self.previous_q, self.previous_qd = time_s, q.copy(), qd
        self.previous_tcp, self.previous_tilt = tcp, tilt
        self.previous_status = status
        self.rows += 1

    def compare_metrics(self, logged, *, complete=True):
        for key, value in self.maxima.items():
            tol = 2e-5 if "acceleration" in key else 8e-7 if "orientation" in key else 1e-7
            if complete:
                close(logged[key], value, "Recomputed motion peak " + key, tol)
            else:
                need(logged[key] + tol >= value, "Failure sidecar loses an already observed motion maximum")


def partition(data):
    need(data["total"] == TOTAL and data["initial_tube"] == SOURCE, "Particle totals differ from 5330/1198")
    for keys, total in ((("beaker", "tube", "airborne", "spilled"), TOTAL),
                        (("tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled"), SOURCE)):
        values = [data[key] for key in keys]
        need(all(type(value) is int and value >= 0 for value in values) and sum(values) == total, "Containment partitions overlap/lose particles")
    need(0 <= data["below_table"] <= data["spilled"], "Invalid below-table subset")


def liquid_measurement(data, *, require_fixed_mass=True):
    partition(data)
    for volume, count in (("received_ml", "tube_to_beaker"), ("remaining_ml", "tube_remaining"),
                          ("inflight_ml", "tube_airborne"), ("spilled_ml", "tube_spilled")):
        close(data[volume], data[count] * QUANTUM, "Label-count water volume " + volume)
    if require_fixed_mass:
        close(data["total_mass_kg"], MASS_TOTAL, "Actual fixed particle mass sum", 1e-12)
    else:
        need(type(data["total_mass_kg"]) in (float, int) and math.isfinite(data["total_mass_kg"])
             and data["total_mass_kg"] > 0, "Invalid recorded mass in failed physical state")


def action_audit(path, record, protocol, kinematic):
    result, conditions = record["result"], record["conditions"]
    footer = effective = None
    raw_rows, evidence_rows, failures = 0, 0, []
    last_step = last_evidence = None
    total_spill = below = 0
    logged_position_violation = 0.
    max_mass_drift = 0.
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        need(header["type"] == "episode", "Missing action header")
        for key in ("target_ml", "seed", "randomized", "policy_params"):
            need(header[key] == conditions[key], "Header conditions differ: " + key)
        close(header["physics_dt_s"], DT, "Header physical dt", 1e-14)
        need(header["particle_count"] == TOTAL and header["minimum_substeps"] == 4
             and header["rendered_training"] is False and header["fluid_mode"] == "live_PBF_action_conditioned", "Unexpected liquid/time/replay mode")
        for name, value in header["source_sha256"].items():
            need(protocol["source_sha256"][PREFIX + name] == value, "Effective header source differs")
        for line in stream:
            row = json.loads(line)
            finite(row)
            need(footer is None, "Records after action result footer")
            kind = row["type"]
            if kind == "effective_fast_sources":
                need(effective is None, "Duplicate effective source metadata")
                effective = row
            elif kind == "step":
                need(row["step"] == raw_rows and not failures, "Missing/reordered action step")
                close(row["physical_time_s"], raw_rows * DT, "Raw action step/60", 1e-12)
                close(row["physics_dt_s"], DT, "Raw action dt", 1e-14)
                truth = row["truth"]
                need(truth["source_partition_count"] == SOURCE, "Source labels not conserved")
                close(truth["mass_kg"], MASS_TOTAL, "Actual logged mass", 1e-12)
                max_mass_drift = max(max_mass_drift, abs(truth["mass_kg"] - MASS_TOTAL))
                need(row["solver_substeps"] >= (4 if raw_rows else 0), "Reduced physical substeps")
                last_step = row
                raw_rows += 1
            elif kind == "fast_step_evidence":
                need(row["step"] == evidence_rows and last_step["step"] == row["step"], "Fast evidence coverage differs")
                close(row["physical_time_s"], row["step"] * DT, "Fast evidence physical clock", 1e-12)
                p = row["partitions"]
                partition(p)
                if row["step"] == 0:
                    need(p["tube_remaining"] == SOURCE and p["tube_to_beaker"] == p["tube_airborne"] == p["tube_spilled"] == 0
                         and p["beaker"] == TOTAL - SOURCE and p["tube"] == SOURCE and p["airborne"] == p["spilled"] == 0,
                         "Reset does not contain the fixed initial source/beaker water labels")
                for volume, count in (("received_ml", "tube_to_beaker"), ("remaining_ml", "tube_remaining"),
                                      ("inflight_ml", "tube_airborne"), ("spilled_ml", "tube_spilled")):
                    close(last_step["truth"][volume], p[count] * QUANTUM, "Action classified source volume")
                status = row["controller_status"]
                need(status["policy_params"] == conditions["policy_params"], "Policy parameters changed within episode")
                close(last_step["source_time"], status["u"], "Actual source path coordinate")
                for key, value in last_step["observation"].items():
                    need(status.get(key) == value, "Raw observation differs from controller status")
                for key, value in last_step["action"].items():
                    need((status.get("policy_action") if key == "policy" else status.get(key)) == value, "Raw action differs from controller status")
                kinematic.observe(row["physical_time_s"], row["kinematics"], status)
                kinematic.compare_metrics(row["fast_metrics"])
                q = row["kinematics"]["joint_position_rad"]
                logged_position_violation = max(logged_position_violation,
                    max(max(lo - value, value - hi, 0.) for value, lo, hi in zip(q, kinematic.path.lower, kinematic.path.upper)))
                total_spill, below = max(total_spill, p["spilled"]), max(below, p["below_table"])
                last_evidence = row
                evidence_rows += 1
            elif kind == "failure":
                failures.append(row)
            elif kind == "result":
                footer = {key: value for key, value in row.items() if key != "type"}
            else:
                raise AuditError("Unknown action trace event " + str(kind))
    need(footer == result and raw_rows == evidence_rows and raw_rows > 0 and effective is not None, "Action footer/evidence mismatch")
    need(effective["source_sha256"] == {name: protocol["source_sha256"][PREFIX + name]
         for name in ("fast_geometry.py", "fast_control.py")}, "Actual injected controller/geometry SHA mismatch")
    need(effective["cache_sha256"] == protocol["input_files"]["path_cache"]["sha256"], "Actual path cache SHA mismatch")
    need(effective["reset_started_utc"] == record["episode_wall_clock_start_utc"], "Explicit pre-reset UTC mismatch")
    utc(effective["reset_started_utc"])
    close(result["physical_time_s"], result["physical_steps"] * DT, "Action footer physical time", 1e-12)
    need(result["physical_steps"] <= 300, "Action exceeds five seconds")
    gap = result["physical_steps"] - last_step["step"]
    need(gap in (0, 1), "Unexplained terminal action coverage gap")
    if failures:
        need(len(failures) == 1 and result["valid_physics"] is False and result["error"] == failures[0]["error"], "Failure event/result mismatch")
        f = failures[0]
        need(f["step"] == last_step["step"] + 1, "Failure event step differs")
        close(f["measurement_physical_time_s"], result["measurement_physical_time_s"], "Failed liquid measurement timestamp")
        liquid_measurement(f["last_available_liquid_measurement"], require_fixed_mass=result["mass_conserved"])
        close(result["received_ml"], f["last_available_liquid_measurement"]["received_ml"], "Failed footer retains actual available liquid truth")
        if f["kinematic_evidence_time_s"] is not None and f["last_motion_evidence"]:
            kinematic.observe(f["kinematic_evidence_time_s"], f["last_motion_evidence"], record["controller_summary"], terminal=True)
        if f["partial_unlogged_fluid_frame_possible"]:
            need(result["measurement_basis"] == "last_complete_measurement_before_partial_fluid_failure", "Partial-fluid failure is not labelled stale measurement")
    else:
        need(gap == 0 and result["error"] is None, "Unexplained unlogged terminal state")
        need(last_evidence["controller_status"] == record["controller_summary"], "Action-end controller sidecar differs")
        close(result["measurement_physical_time_s"], result["physical_time_s"], "Successful liquid measurement clock")
        for key in ("received_ml", "remaining_ml", "inflight_ml", "spilled_ml"):
            close(result[key], last_step["truth"][key], "Action footer vs final physical measurement " + key)
        if result["complete"]:
            window = list(kinematic.home_checks)
            need(len(window) == 13 and all(row["home_static"] and row["airborne"] == 0 for row in window[-12:])
                 and len({row["received_ml"] for row in window}) == 1
                 and record["controller_summary"]["stable_frames"] >= 12,
                 "Completed action lacks twelve physically quiet home frames")
    kinematic.compare_metrics(record["motion_metrics"], complete=not failures)
    for key, expected in (("actual_initial_total_mass_kg", MASS_TOTAL), ("actual_initial_source_mass_kg", MASS_SOURCE)):
        close(result[key], expected, "Initial model mass " + key, 1e-12)
    need(result["particle_count"] == TOTAL and result["source_particle_count"] == SOURCE, "Footer particle count differs")
    close(result["particle_volume_ml"], QUANTUM, "Particle volume quantum", 1e-14)
    close(result["error_ml"], result["received_ml"] - result["target_ml"], "Raw signed error")
    close(result["absolute_error_ml"], abs(result["error_ml"]), "Raw absolute error")
    close(result["declared_path_acceleration_cap"], protocol["path_coordinate"]["collector_path_acceleration_validity_cap"], "V3 own path predicate cap")
    need(result["maximum_mass_drift_kg"] + 1e-12 >= max_mass_drift, "Footer loses mass drift")
    need(result["mass_conserved"] == bool(result["maximum_mass_drift_kg"] <= 1e-12
         and result["particle_mass_array_unchanged"] and result["mass_volume_units_consistent"]), "Mass-conserved flag contradicts recorded model-array checks")
    variation = result["variation"]
    close(variation["initial_jitter_max_spacing"], .03 if conditions["randomized"] else 0., "Initial jitter declaration")
    need(0 <= variation["initial_jitter_realized_max_m"] <= (.03 * .0035 if conditions["randomized"] else 0) + 2 ** -25 + 1e-12,
         "Initial jitter exceeds nominal bound plus float32 position rounding")
    need(.9 <= variation["viscosity_factor"] <= 1.1, "Viscosity randomization out of range")
    if conditions["randomized"]:
        rng = kinematic.np.random.default_rng(int(conditions["seed"]))
        rng.uniform(-.03, .03, (TOTAL, 3))
        expected_viscosity_factor = float(rng.uniform(.9, 1.1))
    else:
        expected_viscosity_factor = 1.
    close(variation["viscosity_factor"], expected_viscosity_factor, "Seeded randomization reproduces independently", 1e-12)
    close(variation["dynamic_viscosity_pa_s"], .001 * variation["viscosity_factor"], "Actual randomized viscosity", 1e-12)
    close(variation["kinematic_viscosity_m2_s"], variation["dynamic_viscosity_pa_s"] / 1000, "Kinematic viscosity", 1e-14)
    expected_valid = bool(result["error"] is None and result["all_states_finite"] and result["source_partition_conserved"]
        and result["mass_conserved"] and result["reset_max_position_error_m"] < 1e-7
        and result["max_source_rate"] <= 1.00001
        and result["max_source_acceleration"] <= result["declared_path_acceleration_cap"] + 1e-5 and result["minimum_substeps"] >= 4)
    need(result["valid_physics"] == expected_valid, "Generated V3 validity flag contradicts its declared predicate")
    return {"raw_step_rows": raw_rows, "unlogged_terminal_failure_steps": gap, "failure_events": failures,
            "maximum_joint_position_violation_rad": float(logged_position_violation),
            "maximum_total_spilled_particles": total_spill, "maximum_below_table_particles": below,
            "last_recorded_partitions": last_evidence["partitions"], "reset_started_utc": effective["reset_started_utc"],
            "footer_equals_original_returned_result": True}


def post_audit(path, record, kinematic):
    result, post = record["result"], record["post_observation"]
    if path is None:
        need(post is None or post.get("error"), "Post summary without raw observations")
        return None if post is None else {"rows": 0, "partial": True, "missing_before_first_observation": True,
            "raw_check": {"raw_rows": 0, "partial_after_exception": True, "exception_before_first_observation": True,
                "physical_seconds_logged": 0., "deadline_received_ml": result["received_ml"] if abs(result["physical_time_s"] - 5) < 1e-10 else None,
                "maximum_total_spilled_particles": 0, "all_logged_states_finite": None, "all_logged_mass_arrays_unchanged": None}}
    count, maximum_change, maximum_total_spill, maximum_below = 0, 0., 0, 0
    min_total = min_source = None
    max_source = 0
    all_finite = all_mass = True
    deadline_dose = result["received_ml"] if abs(result["physical_time_s"] - 5) < 1e-10 else None
    last = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            finite(row)
            count += 1
            need(row["type"] == "post_observation" and row["step"] == count, "Post observation sequence differs")
            close(row["physical_time_s"], result["physical_time_s"] + count * DT, "Post physical time", 1e-12)
            liquid_measurement(row["measurement"], require_fixed_mass=bool(post.get("mass_array_unchanged")))
            need(row["solver_substeps"] >= 4, "Post-observation solver substeps reduced")
            kinematic.observe(row["physical_time_s"], row["kinematics"], row["controller_status"])
            maximum_change = max(maximum_change, abs(row["measurement"]["received_ml"] - result["received_ml"]))
            maximum_total_spill = max(maximum_total_spill, row["measurement"]["spilled"])
            maximum_below = max(maximum_below, row["measurement"]["below_table"])
            min_total = row["measurement"]["spilled"] if min_total is None else min(min_total, row["measurement"]["spilled"])
            max_source = max(max_source, row["measurement"]["tube_spilled"])
            min_source = row["measurement"]["tube_spilled"] if min_source is None else min(min_source, row["measurement"]["tube_spilled"])
            all_finite = all_finite and row["all_finite"] is True
            all_mass = all_mass and row["mass_array_unchanged"] is True
            if abs(row["physical_time_s"] - 5) < 1e-10:
                deadline_dose = row["measurement"]["received_ml"]
            last = row
    failed = bool(post.get("error"))
    if not failed:
        need(count == round((7 - result["physical_time_s"]) / DT) and last is not None, "Post observation does not reach t7")
        close(post["start_physical_time_s"], result["physical_time_s"], "Post actual start")
        close(post["duration_s"], count * DT, "Post duration")
        close(post["end_physical_time_s"], 7., "Post actual end")
        need(post["duration_s"] >= 2 - 1e-10 and post["deadline_received_ml"] == deadline_dose, "Missing/mismatched exact-5s measurement")
        close(post["maximum_received_change_ml"], maximum_change, "Late-dose maximum change")
        close(post["final_received_ml"], last["measurement"]["received_ml"], "Final post dose")
        close(post["final_spilled_ml"], last["measurement"]["spilled_ml"], "Final post source spill")
        need(post["all_states_finite"] == all_finite and post["mass_array_unchanged"] == all_mass, "Post finite/mass flags differ from raw observations")
        expected_pass = maximum_change == 0 and post["all_states_finite"] and post["mass_array_unchanged"] and post["final_spilled_ml"] == 0
        need(post["passed"] == expected_pass, "Original late-observation task flag differs")
    elif record["motion_evidence"].get("post_terminal_kinematics"):
        kinematic.terminal_without_status(record["motion_evidence"]["post_terminal_kinematics"])
    kinematic.compare_metrics(record["motion_evidence"]["post_motion_metrics"], complete=not failed)
    return {"rows": count, "partial": failed, "deadline_received_ml": deadline_dose,
            "maximum_received_change_ml": maximum_change, "maximum_total_spilled_particles": maximum_total_spill,
            "maximum_below_table_particles": maximum_below,
            "raw_check": {"raw_rows": count, "partial_after_exception": failed, "physical_seconds_logged": count * DT,
                "deadline_received_ml": deadline_dose, "maximum_total_spilled_particles": maximum_total_spill,
                "minimum_total_spilled_particles": min_total, "maximum_source_spilled_particles": max_source,
                "minimum_source_spilled_particles": min_source, "maximum_below_table_particles": maximum_below,
                "all_logged_states_finite": all_finite, "all_logged_mass_arrays_unchanged": all_mass}}


def assessed(record):
    """Independently derive the frozen runner's unchanged formal task criteria."""
    result, status = record["result"], record["controller_summary"]
    motion, post_motion = record["motion_metrics"], record["motion_evidence"]["post_motion_metrics"]
    post, check, post_check = record["post_observation"], record["trace_check"], record["post_trace_check"]
    def safe_motion(m):
        return (m["maximum_joint_speed_limit_ratio"] <= 1.00001 and m["maximum_joint_acceleration_rad_s2"] <= 25.0001
                and m["maximum_hand_fk_position_error_m"] <= .003 and m["maximum_hand_fk_orientation_error_rad"] <= .02)
    safe = bool(result["valid_physics"] and result["mass_conserved"] and result["all_states_finite"]
        and result["source_partition_conserved"] and result["max_spilled_ml"] == 0 and result["max_below_table_particles"] == 0
        and safe_motion(motion) and safe_motion(post_motion) and check["maximum_joint_position_violation_rad"] <= 1e-6
        and check["maximum_total_spilled_particles"] == check["maximum_below_table_particles"] == 0
        and (post_check is None or (post_check["maximum_total_spilled_particles"] == 0
            and post_check["all_logged_states_finite"] and post_check["all_logged_mass_arrays_unchanged"]))
        and not (isinstance(post, dict) and post.get("error")))
    home = bool(status.get("home_static") is True and status.get("home_position_error_m", math.inf) <= .001
        and status.get("home_orientation_error_rad", math.inf) <= math.radians(.1)
        and status.get("home_joint_error_rad", math.inf) <= .01 and abs(status.get("u", math.inf)) <= 1e-7
        and status.get("source_airborne_particles") == 0 and status.get("stable_frames", -1) >= 12)
    on_time = bool(result["complete"] and result["physical_time_s"] <= 5 + 1e-12 and result["physical_steps"] <= 300
        and status.get("completed_by_deadline") is True and status.get("timed_out") is False and status.get("done") is True and home)
    sample = post.get("deadline_received_ml") if isinstance(post, dict) else None
    available = type(sample) in (int, float) and math.isfinite(sample)
    late = bool(isinstance(post, dict) and post.get("passed") is True and available and post.get("duration_s", -1) >= 2 - 1e-10
        and post.get("end_physical_time_s", -1) >= 7 - 1e-10
        and abs(post.get("start_physical_time_s", math.inf) - result["physical_time_s"]) < 1e-10
        and post.get("maximum_received_change_ml") == 0 and post.get("all_states_finite") is True and post.get("mass_array_unchanged") is True)
    error = sample - result["target_ml"] if available else result["error_ml"]
    return {"physical_safe": safe, "home_stable": home, "completed_by_deadline": on_time, "late_drop_passed": late,
            "constraint_feasible": bool(safe and on_time and late), "dose_error_ml": error, "absolute_error_ml": abs(error),
            "deadline_dose_measurement_available": available,
            "dose_measurement_basis": "physical_5s" if available else "invalid_terminal_fallback_not_a_5s_measurement",
            "fallback_measurement_physical_time_s": result.get("measurement_physical_time_s") if not available else None,
            "fallback_measurement_basis": result.get("measurement_basis") if not available else None,
            "action_physical_seconds": result["physical_time_s"],
            "success_within_0_1_ml": bool(safe and on_time and late and abs(error) <= .1),
            "action_result_was_not_modified": True, "action_footer_success_alone_is_not_V3_release_success": True}


def metrics(records):
    n = len(records)
    errors = [record["assessment"]["absolute_error_ml"] for record in records]
    return {"episodes": n, "mae_ml": sum(errors) / n, "worst_absolute_error_ml": max(errors),
        "unsafe_count": sum(not r["assessment"]["physical_safe"] for r in records),
        "deadline_or_home_or_late_drop_failures": sum(not (r["assessment"]["completed_by_deadline"] and r["assessment"]["late_drop_passed"]) for r in records),
        "feasible_count": sum(r["assessment"]["constraint_feasible"] for r in records),
        "missing_deadline_measurements": sum(not r["assessment"]["deadline_dose_measurement_available"] for r in records),
        "within_0_1_ml_count": sum(r["assessment"]["success_within_0_1_ml"] for r in records),
        "within_0_2_ml_count": sum(r["assessment"]["constraint_feasible"] and e <= .2 for r, e in zip(records, errors)),
        "all_mass_conserved": all(r["result"]["mass_conserved"] for r in records),
        "mean_action_physical_seconds": sum(r["result"]["physical_time_s"] for r in records) / n,
        "physical_steps": sum(r["result"]["physical_steps"] for r in records),
        "action_physical_seconds": sum(r["result"]["physical_time_s"] for r in records),
        "post_observation_physical_seconds": sum(r["post_observation"].get("duration_s", 0) for r in records if isinstance(r["post_observation"], dict))}


def score(records):
    m = metrics(records)
    groups = [records] + [[r for r in records if r["conditions"]["target_ml"] == target] for target in (20, 30, 40)]
    violation = sum(max(0., metrics(group)["mae_ml"] - .1) + max(0., metrics(group)["worst_absolute_error_ml"] - .2) for group in groups if group)
    return (m["unsafe_count"], m["deadline_or_home_or_late_drop_failures"], violation,
            m["mae_ml"] + .5 * m["worst_absolute_error_ml"], m["mean_action_physical_seconds"])


def released(records):
    groups = [records] + [[r for r in records if r["conditions"]["target_ml"] == target] for target in (20, 30, 40)]
    return bool(all(groups) and all(metrics(g)["unsafe_count"] == 0 and metrics(g)["deadline_or_home_or_late_drop_failures"] == 0
        and metrics(g)["mae_ml"] <= .1 and metrics(g)["worst_absolute_error_ml"] <= .2 and metrics(g)["all_mass_conserved"] for g in groups))


def verify_manifests(run, root, summary):
    document, frozen = read(run / "protocol.json"), read(run / "input_freeze.json")
    p, d = document["protocol"], frozen["declaration"]
    need(identity(p) == document["protocol_id"] == summary["protocol_id"] and summary["protocol"] == p, "Protocol identity mismatch")
    need(identity(d) == frozen["input_id"] == p["input_id"], "Input freeze identity mismatch")
    for key, value in d.items():
        need(p[key] == value, "Protocol changes frozen input " + key)
    need(p["budget"] == {"generations": 3, "population": 6, "elites": 2, "training": 108, "validation": 12, "test": 18, "total": 138}, "Not the complete declared 138 budget")
    need(p["targets_ml"] == [20, 30, 40] and p["train_seeds"] == [61101, 61102]
         and p["validation_seeds"] == [62101, 62102] and p["test_seeds"] == [71101, 71102, 71103], "Seed/target declaration differs")
    need(p["neural_policy"] is False and p["ppo"] is False and p["automatic_gui_deployment"] is False, "Algorithm/deployment mislabeled")
    constants = source_constants(run / "source_snapshot" / PREFIX / "fast_control.py")
    need(p["cem"]["baseline_params"] == constants["DEFAULT_PARAMS"]
         and p["parameter_units"] == constants["PARAMETER_UNITS"]
         and p["cem"]["parameter_names"] == list(constants["DEFAULT_PARAMS"]), "Frozen parameter identity/units differ from source")
    for name, bounds in p["cem"]["bounds"].items():
        lo, hi = constants["PARAMETER_BOUNDS"][name]
        need(lo <= bounds[0] < bounds[1] <= hi, "Search bounds exceed controller bounds")
        need(bounds[0] <= p["cem"]["initial_params"][name] <= bounds[1], "Initial CEM mean outside frozen search bounds")
    physics = p["physics"]
    need(physics["cuda_graph"] is False and physics["minimum_substeps"] == 4 and physics["pressure_iterations"] == 6, "Physics dispatch/iterations changed")
    for key, expected in (("dt_s", DT), ("action_deadline_s", 5.), ("post_observation_until_physical_time_s", 7.), ("tolerance_ml", .1), ("maximum_absolute_error_ml", .2)):
        close(physics[key], expected, "Protocol physics " + key)
    need(p["completion"]["stable_frames"] == 12 and p["robot_envelope"]["joint_derivative_sampling_hz"] == 240
         and p["robot_envelope"]["hand_fk_and_tcp_sampling_hz"] == 60, "Completion/kinematic sampling changed")
    need(len(p["source_sha256"]) == 20 and len(p["protected_legacy_source_sha256"]) == 11, "Source guard coverage differs")
    for name, expected in p["source_sha256"].items():
        need(digest(inside(root, name)) == digest(inside(run / "source_snapshot", name)) == expected, "Frozen/current source differs: " + name)
    for name, entry in p["input_files"].items():
        need(digest(entry["path"]) == entry["sha256"], "Current immutable raw input changed: " + name)
        snap = run / "input_snapshot" / (name + Path(entry["path"]).suffix)
        need(digest(snap) == entry["sha256"], "Raw input snapshot changed: " + name)
    if p["historical_manifest_verified"]:
        old = root / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated/protocol.json"
        need(digest(old) == p["historical_manifest_sha256"]
             and read(old)["protocol"]["source_sha256"] == p["protected_legacy_source_sha256"], "Protected historical eleven mismatch")
    else:
        need(p["historical_manifest_sha256"] is None, "Unverified history claims a digest")
    def integer_values(value):
        if type(value) is int:
            return {value}
        if isinstance(value, (list, dict)):
            children = value.values() if isinstance(value, dict) else value
            result = set()
            for child in children:
                result.update(integer_values(child))
            return result
        return set()
    def seeds_in(value):
        result = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if "seed" in key.lower():
                    result.update(integer_values(child))
                result.update(seeds_in(child))
        elif isinstance(value, list):
            for child in value:
                result.update(seeds_in(child))
        return result
    all_previous_seeds = set()
    for name, expected in p["historical_seed_protocol_sha256"].items():
        historical_path = inside(root, name)
        need(digest(historical_path) == expected, "Preexisting seed manifest changed")
        all_previous_seeds.update(seeds_in(read(historical_path)))
    splits = [set(p[key]) for key in ("train_seeds", "validation_seeds", "test_seeds")]
    need(all(splits[i].isdisjoint(splits[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
         and all_previous_seeds.isdisjoint(set.union(*splits)), "Initial-state seeds leak across split/history")
    gui = {path.relative_to(root).as_posix(): digest(path) for path in sorted((root / "assets/robot_testtube_pour/policy").glob("*.json"))}
    need(gui == paths(p["gui_checkpoint_sha256_before"]) == paths(summary["gui_checkpoint_sha256_after"]), "Protected GUI checkpoints changed")
    need(p["release"] == {"all_conditions_within_5s_and_home_stable_and_no_late_drop": True,
         "all_safe_mass_conserved": True, "mae_ml_at_most": .1, "worst_absolute_error_ml_at_most": .2,
         "per_target_thresholds_also_apply": True}, "Formal task release requirements were relaxed")
    gate = p["engineering_gate"]
    need(gate["CEM_elite_and_release_zero_spill_requirements_relaxed"] is False
         and gate["engineering_gate_does_not_certify_task_success"] is True
         and gate["allow_existing_engineering_spill_for_optimization"] == p["allow_engineering_spill"], "Engineering permission silently changes task success")
    for kind in ("action", "post"):
        for target, entry in gate[f"raw_{kind}_trace_evidence"].items():
            need(entry["sha256"] == p["input_files"][f"engineering_{kind}_t" + target]["sha256"], "Engineering raw trace binding differs")
    return p, document["protocol_id"], frozen


def episode_audit(run, record, protocol, path):
    finite(record)
    for path_key in ("record_path", "motion_record_path", "post_record_path"):
        if record.get(path_key) is not None:
            need(digest(inside(run, record[path_key])) == record[path_key.replace("_path", "_sha256")], "Episode artifact SHA mismatch: " + record["episode_id"])
    need(read(run / record["motion_record_path"]) == record["motion_evidence"], "Motion sidecar differs from journal")
    evidence = record["motion_evidence"]
    need(evidence["action_motion_metrics"] == record["motion_metrics"]
         and evidence["controller_at_action_end"] == record["controller_summary"]
         and evidence["post_observation"] == record["post_observation"] and evidence["action_result_modified"] is False,
         "Motion/action/post data were mixed or rewritten")
    need(evidence["episode_id"] == record["episode_id"] and evidence["protocol_id"] == record["protocol_id"], "Foreign motion sidecar")
    kinematic = KinematicAudit(path)
    action = action_audit(run / record["record_path"], record, protocol, kinematic)
    post = post_audit(run / record["post_record_path"] if record["post_record_path"] else None, record, kinematic)
    equal(record["trace_check"], action, "Frozen runner trace check vs independent raw evidence")
    equal(record["post_trace_check"], post["raw_check"] if post is not None else None, "Frozen runner post check vs raw observations")
    equal(evidence["trace_check"], record["trace_check"], "Sidecar raw action check")
    equal(evidence["post_trace_check"], record["post_trace_check"], "Sidecar raw post check")
    equal(record["assessment"], assessed(record), "Recomputed formal assessment")
    # Unlike the frozen task predicate, these independent observations also
    # expose any discrepancy between actual raw kinematics and reported flags.
    independent_motion = (kinematic.max_position_violation <= 1e-6
        and kinematic.maxima["maximum_joint_speed_limit_ratio"] <= 1.00001
        and kinematic.maxima["maximum_joint_acceleration_rad_s2"] <= 25.0001
        and kinematic.maxima["maximum_hand_fk_position_error_m"] <= .003
        and kinematic.maxima["maximum_hand_fk_orientation_error_rad"] <= .0200006)
    return {"episode_id": record["episode_id"], "stage": record["conditions"]["stage"],
        "role": record["conditions"]["policy_role"], "target_ml": record["conditions"]["target_ml"], "seed": record["conditions"]["seed"],
        "action": action, "post": post, "independent_motion_checks_passed": bool(independent_motion),
        "recomputed_motion_peaks": kinematic.maxima, "joint_240hz_rows": kinematic.dense_rows,
        "fk_tcp_60hz_rows": kinematic.rows, "maximum_command_integration_residual": kinematic.command_residual,
        "failed_post_terminal_statuses_inferred_from_prior_command": kinematic.inferred_terminal_statuses,
        "assessment": record["assessment"], "measurement_basis": record["result"]["measurement_basis"],
        "measurement_physical_time_s": record["result"]["measurement_physical_time_s"],
        "original_valid_physics": record["result"]["valid_physics"]}


def engineering_audit(run, protocol, path):
    """Recheck the pre-training exception without applying it to task release."""
    report = read(run / "input_snapshot/engineering_report.json")
    declared = read(run / "input_snapshot/engineering_protocol.json")
    gate = protocol["engineering_gate"]
    need(report["complete"] is True and declared["purpose"] == "engineering_feasibility_not_heldout"
         and declared["policy_parameters"] == protocol["cem"]["baseline_params"], "Engineering profile/model mismatch")
    need(len(report["results"]) == 3 and {row["result"]["target_ml"] for row in report["results"]} == {20, 30, 40}, "Engineering target coverage differs")
    defects, rows = [], []
    for original in report["results"]:
        target = str(int(original["result"]["target_ml"]))
        action_file = run / "input_snapshot" / ("engineering_action_t" + target + ".gz")
        post_file = run / "input_snapshot" / ("engineering_post_t" + target + ".gz")
        with gzip.open(action_file, "rt", encoding="utf-8") as stream:
            header, effective = json.loads(next(stream)), json.loads(next(stream))
        record = {"conditions": {key: header[key] for key in ("target_ml", "seed", "randomized", "policy_params")},
            "result": original["result"], "controller_summary": original["controller"],
            "motion_metrics": original["motion_metrics"], "motion_evidence": {"post_motion_metrics": original["motion_metrics"]},
            "post_observation": original["post_observation"], "episode_wall_clock_start_utc": effective["reset_started_utc"]}
        need(header["policy_params"] == protocol["cem"]["baseline_params"], "Engineering baseline params changed")
        kin = KinematicAudit(path)
        action = action_audit(action_file, record, protocol, kin)
        post = post_audit(post_file, record, kin)
        equal(action, gate["raw_action_trace_evidence"][target]["verification"], "Engineering action provenance")
        equal(post["raw_check"], gate["raw_post_trace_evidence"][target]["verification"], "Engineering post provenance")
        result, status = original["result"], original["controller"]
        need(result["valid_physics"] and result["mass_conserved"] and result["complete"]
             and result["physical_time_s"] <= 5 and status["completed_by_deadline"] and not status["timed_out"]
             and status["home_static"] and status["stable_frames"] >= 12, "Engineering baseline is not complete within five seconds")
        need(not post["partial"] and post["maximum_received_change_ml"] == 0
             and post["maximum_below_table_particles"] == action["maximum_below_table_particles"] == 0,
             "Engineering observation adds late-dose/below-table defects")
        end = action["last_recorded_partitions"]
        raw = post["raw_check"]
        need(raw["maximum_total_spilled_particles"] == raw["minimum_total_spilled_particles"] == end["spilled"]
             and raw["maximum_source_spilled_particles"] == raw["minimum_source_spilled_particles"] == end["tube_spilled"],
             "Engineering post adds/changes existing spill counts")
        if result["max_spilled_ml"] > 0 or action["maximum_total_spilled_particles"] > 0:
            need(protocol["allow_engineering_spill"] is True, "Existing engineering spill was not explicitly authorized")
            defects.append({"target_ml": result["target_ml"], "action_end_source_spilled_ml": result["spilled_ml"],
                "action_max_source_spilled_ml": result["max_spilled_ml"], "action_end_source_spilled_particles": end["tube_spilled"],
                "action_end_total_spilled_particles": end["spilled"], "action_max_total_spilled_particles": action["maximum_total_spilled_particles"],
                "deadline_received_ml": post["deadline_received_ml"], "deadline_dose_error_ml": post["deadline_received_ml"] - result["target_ml"],
                "original_post_passed": original["post_observation"]["passed"], "post_spill_counts_unchanged": True,
                "task_zero_spill_requirement_passed": False})
        rows.append({"target_ml": result["target_ml"], "action_seconds": result["physical_time_s"],
                     "post_rows": post["rows"], "joint_240hz_rows": kin.dense_rows})
    equal(defects, gate["initial_task_spill_defects"], "All initial engineering spill defects retained")
    return {"numerical_and_kinematic_feasibility_verified": True, "existing_spill_defects": defects,
            "engineering_exception_is_not_task_success": True, "formal_zero_spill_criteria_unchanged": True, "cases": rows}


def replay_cem(run, records, p, pid):
    config = p["cem"]
    rng = random.Random(config["seed"])
    names, bounds = config["parameter_names"], config["bounds"]
    mean = dict(config["initial_params"])
    std = {key: (bounds[key][1] - bounds[key][0]) * config["initial_std_fraction"] for key in names}
    best_params = best_key = None
    generations, train = [], []
    for generation in range(3):
        candidates = [dict(mean)]
        if best_params is not None and best_params != mean:
            candidates.append(dict(best_params))
        while len(candidates) < 6:
            candidates.append({key: min(bounds[key][1], max(bounds[key][0], rng.gauss(mean[key], std[key]))) for key in names})
        groups, keys = [], []
        for index, params in enumerate(candidates):
            group = [records[f"train_g{generation:03d}_c{index:03d}_t{target}_s{seed}"] for target in p["targets_ml"] for seed in p["train_seeds"]]
            need(all(record["conditions"]["policy_params"] == params for record in group), "CEM sampled parameters do not reproduce from frozen seed")
            groups.append(group)
            keys.append(score(group))
            train.extend(group)
        indices = sorted(range(6), key=lambda index: (keys[index], index))
        for index in indices:
            if best_key is None or keys[index] < best_key:
                best_key, best_params = keys[index], dict(candidates[index])
        for name in names:
            elite_mean = sum(candidates[index][name] for index in indices[:2]) / 2
            elite_std = math.sqrt(sum((candidates[index][name] - elite_mean) ** 2 for index in indices[:2]) / 2)
            mean[name] = .65 * mean[name] + .35 * elite_mean
            std[name] = max(.02 * (bounds[name][1] - bounds[name][0]), .65 * std[name] + .35 * elite_std)
        update = {"elite_indices": indices[:2], "generation_best_key": list(keys[indices[0]]), "best_key": list(best_key),
                  "best_params": dict(best_params), "next_mean": dict(mean), "next_std": dict(std)}
        entry = {"generation": generation, "candidates": [{"index": index, "params": params,
            "selection_key": list(key), "metrics": metrics(group), "episode_ids": [record["episode_id"] for record in group]}
            for index, (params, key, group) in enumerate(zip(candidates, keys, groups))],
            "completed_training_episodes": len(train), **update}
        equal(read(run / f"generation_{generation:03d}.json"), {"protocol_id": pid, **entry}, "Generation " + str(generation))
        generations.append(entry)
    equal(read(run / "search_state.json"), {"protocol_id": pid, "generations_completed": 3, **update}, "Final CEM state")
    return generations, train, best_params, best_key


def audit_run(run=DEFAULT_RUN, *, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    # Never inspect active heldout outcomes. This is deliberately the first IO
    # involving trial results, before journal/models/cache dependency imports.
    summary_file = run / "summary.json"
    need(summary_file.is_file(), "Refusing partial run: final summary.json is absent")
    summary = read(summary_file)
    need(summary.get("complete") is True and summary.get("total_recorded_episodes") == 138, "Refusing partial run: all 138 episodes must be complete")
    finite(summary)
    p, pid, frozen = verify_manifests(run, root, summary)
    records, order = {}, []
    with (run / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            name, condition = record["episode_id"], record["conditions"]
            need(name not in records and record["protocol_id"] == pid and identity(condition) == record["conditions_sha256"], "Duplicate/foreign/damaged journal record")
            need(record["record_path"] == f"traces/{name}.steps.jsonl.gz"
                 and record["motion_record_path"] == f"motion/{name}.json"
                 and record["post_record_path"] in (None, f"traces/{name}.post.jsonl.gz"),
                 "Episode reuses a foreign/noncanonical trace or sidecar")
            records[name], order = record, order + [name]
    expected = {f"train_g{g:03d}_c{c:03d}_t{t}_s{s}" for g in range(3) for c in range(6) for t in p["targets_ml"] for s in p["train_seeds"]}
    expected |= {f"{stage}_{role}_t{t}_s{s}" for stage, seeds in (("validation", p["validation_seeds"]), ("test", p["test_seeds"]))
                 for role in ("candidate", "baseline") for t in p["targets_ml"] for s in seeds}
    need(len(records) == 138 and set(records) == expected, "Journal is missing/adding declared episodes")
    expected_raw = {r[key] for r in records.values() for key in ("record_path", "post_record_path") if r[key] is not None}
    need({f.relative_to(run).as_posix() for f in (run / "traces").glob("*.gz")} == expected_raw,
         "Unjournaled/missing raw gzip artifacts remain in the completed trace directory")
    need(Counter(record["conditions"]["stage"] for record in records.values()) == {"train": 108, "validation": 12, "test": 18}, "Stage counts differ")
    for g in range(3):
        for c in range(6):
            for target in p["targets_ml"]:
                for seed in p["train_seeds"]:
                    record = records[f"train_g{g:03d}_c{c:03d}_t{target}_s{seed}"]
                    for key, value in {"stage": "train", "policy_role": "candidate", "target_ml": float(target),
                        "seed": seed, "generation": g, "candidate": c, "randomized": True}.items():
                        need(record["conditions"][key] == value, "Training journal condition differs from ID: " + key)
    for stage, seeds in (("validation", p["validation_seeds"]), ("test", p["test_seeds"])):
        for role in ("candidate", "baseline"):
            for target in p["targets_ml"]:
                for seed in seeds:
                    record = records[f"{stage}_{role}_t{target}_s{seed}"]
                    for key, value in {"stage": stage, "policy_role": role, "target_ml": float(target), "seed": seed,
                        "generation": None, "candidate": None, "randomized": True}.items():
                        need(record["conditions"][key] == value, "Validation/test condition differs from ID: " + key)
    for record in records.values():
        need(record["result"]["policy_params"] == record["conditions"]["policy_params"], "Footer changed policy params")
        need((record["result"]["target_ml"], record["result"]["seed"], record["result"]["randomized"])
             == (record["conditions"]["target_ml"], record["conditions"]["seed"], True), "Footer conditions mismatch")
        need(utc(record["completed_at_utc"]) >= utc(record["episode_wall_clock_start_utc"]), "Episode UTC ordering invalid")
    with (run / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    need(len(csv_rows) == 138 and {row["episode_id"] for row in csv_rows} == set(records), "CSV drops/duplicates episodes")
    for row in csv_rows:
        record = records[row["episode_id"]]
        close(float(row["dose_error_at_deadline_ml"]), record["assessment"]["dose_error_ml"], "CSV deadline/fallback dose")
        close(float(row["raw_action_footer_error_ml"]), record["result"]["error_ml"], "CSV original action dose")
    cache_path = FrozenPath(run, p, root)
    engineering = engineering_audit(run, p, cache_path)
    reports = [episode_audit(run, records[name], p, cache_path) for name in order]
    generations, training, candidate, best_key = replay_cem(run, records, p, pid)
    with (run / "training_curve.csv").open(encoding="utf-8-sig", newline="") as stream:
        curve = list(csv.DictReader(stream))
    need(len(curve) == 3, "Training curve invents/omits generations")
    for index, row in enumerate(curve):
        close(float(row["generation"]), index, "Training curve generation")
        close(float(row["completed_training_episodes"]), (index + 1) * 36, "Training curve episode count")
        for component, name in enumerate(p["selection_tuple"]):
            values = [candidate_row["selection_key"][component] for candidate_row in generations[index]["candidates"]]
            close(float(row["mean_" + name]), sum(values) / 6, "CEM curve mean component")
            close(float(row["best_" + name]), generations[index]["best_key"][component], "CEM curve seen-best component")
    equal(summary["training"], {"generations": generations, "metrics": metrics(training), "best_key": list(best_key),
        "best_params": candidate, "parameters_changed_from_initial_baseline": candidate != p["cem"]["baseline_params"],
        "distinct_parameter_vectors": len({identity(row["params"]) for generation in generations for row in generation["candidates"]})}, "Training summary")
    validation = {role: [records[f"validation_{role}_t{target}_s{seed}"] for seed in p["validation_seeds"] for target in p["targets_ml"]] for role in ("candidate", "baseline")}
    baseline = p["cem"]["baseline_params"]
    for role, params in (("candidate", candidate), ("baseline", baseline)):
        need(all(r["conditions"]["policy_params"] == params for r in validation[role]), "Validation policy parameters differ")
    keys = {role: score(group) for role, group in validation.items()}
    recommended = "candidate" if candidate != baseline and keys["candidate"] < keys["baseline"] else "baseline"
    validation_report = {"protocol_id": pid, "selection_keys": {role: list(key) for role, key in keys.items()},
        "metrics": {role: metrics(group) for role, group in validation.items()},
        "episode_ids": {role: [r["episode_id"] for r in group] for role, group in validation.items()},
        "recommended_role": recommended, "test_used_for_selection": False}
    equal(read(run / "validation.json"), validation_report, "Validation report")
    equal(summary["validation"], validation_report, "Validation summary")
    freeze = read(run / "freeze.json")
    recommended_params = candidate if recommended == "candidate" else baseline
    need(freeze["protocol_id"] == pid and freeze["candidate_params"] == candidate
         and freeze["recommended_params"] == recommended_params and freeze["recommended_role"] == recommended
         and freeze["test_used_for_selection"] is False and freeze["automatic_gui_deployment"] is False, "Frozen selection differs")
    for role, params in (("candidate", candidate), ("recommended", recommended_params)):
        model_file = run / (role + "_frozen.json")
        need(digest(model_file) == freeze[role + "_sha256"], "Frozen model SHA changed")
        model = read(model_file)
        need(model["parameters"] == params and model["deadline_physical_seconds"] == 5, "Frozen model kind/parameters/deadline mismatch")
        need(model["schema_version"] == 1 and model["policy_kind"] == source_constants(run / "source_snapshot" / PREFIX / "fast_control.py")["POLICY_KIND"], "Wrong frozen policy class")
        metadata = model["metadata"]
        need(metadata["protocol_id"] == pid and metadata["frozen_at_utc"] == freeze["frozen_at_utc"]
             and metadata["source_sha256"] == p["source_sha256"] and metadata["test_used_for_selection"] is False
             and metadata["automatic_gui_deployment"] is False and metadata["neural_policy"] is False, "Frozen model metadata contradicts isolation")
        need(metadata["validation_selection_keys"] == {key: list(value) for key, value in keys.items()}
             and metadata["parameters_changed_from_initial_baseline"] == (candidate != baseline)
             and metadata["role"] == role and metadata["recommended_role"] == recommended,
             "Frozen validation/model-change metadata differs")
        equal(summary["models"][role], {"file": model_file.name, "sha256": freeze[role + "_sha256"], "parameters": params}, "Model summary")
    need((run / "models_frozen.sha256").read_text(encoding="utf-8").splitlines()
         == [freeze[role + "_sha256"] + "  " + role + "_frozen.json" for role in ("candidate", "recommended")], "Two-model checksum manifest changed")
    first_test = min(utc(record["episode_wall_clock_start_utc"]) for record in records.values() if record["conditions"]["stage"] == "test")
    need(utc(freeze["frozen_at_utc"]) < first_test, "Holdout started before two-model freezing")
    need(max(utc(record["completed_at_utc"]) for record in records.values() if record["conditions"]["stage"] != "test") <= utc(freeze["frozen_at_utc"]), "Train/validation completed after model freeze")
    pairs, test = [], {"candidate": [], "baseline": []}
    for seed in p["test_seeds"]:
        for target in p["targets_ml"]:
            pair = {"target_ml": target, "seed": seed, "recommended_result_role": recommended}
            for role, params in (("candidate", candidate), ("baseline", baseline)):
                record = records[f"test_{role}_t{target}_s{seed}"]
                need(record["conditions"]["policy_params"] == params, "Holdout changed frozen parameters")
                pair[role] = record
                test[role].append(record)
            pairs.append(pair)
    expected_test = {"paired_results": pairs, "metrics": {role: metrics(group) for role, group in test.items()},
                     "distinct_target_seed_conditions": 9, "executed_live_episodes": 18, "recommended_metrics": metrics(test[recommended])}
    equal(summary["test"], expected_test, "Holdout summary including every failed case")
    release = {role + "_passed": released(group) for role, group in (*test.items(), ("recommended", test[recommended]))}
    release.update(does_not_reselect_model=True, automatic_gui_deployment=False)
    equal(summary["release_acceptance"], release, "Original unchanged release gates")
    equal(summary["recommendation"], {"role": recommended, "selected_on_validation_only": True, "additional_test_rollouts": 0}, "Validation-only recommendation")
    equal(summary["all_rollouts"], metrics(list(records.values())), "All rollouts, no outlier filtering")
    need(summary["raw_action_result_footers_modified"] is False and summary["test_used_for_training_or_selection"] is False, "Result rewriting or test fitting declared")
    need(summary["all_episode_ids"] == order, "Summary episode list drops/reorders journal")
    diagnostics = [row for row in reports if not row["assessment"]["success_within_0_1_ml"] or not row["independent_motion_checks_passed"]]
    return {"audit_schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_tool_sha256": digest(__file__), "artifact_integrity_verified": True, "protocol_id": pid,
        "summary_sha256": digest(summary_file), "journal_sha256": digest(run / "episodes.jsonl"),
        "budgets": {"training": 108, "validation": 12, "test": 18, "total": 138}, "episodes_audited": 138,
        "source_files_and_snapshots_verified": 20, "protected_legacy_sources_verified": 11,
        "raw_action_trace_count": len(reports), "raw_post_trace_count": sum(record["post_record_path"] is not None for record in records.values()),
        "all_failures_retained": True, "raw_action_result_footers_unchanged": True,
        "joint_240hz_samples_audited": sum(row["joint_240hz_rows"] for row in reports),
        "fk_tcp_60hz_rows_audited": sum(row["fk_tcp_60hz_rows"] for row in reports),
        "test_metrics": {**{role: metrics(group) for role, group in test.items()}, "recommended": metrics(test[recommended])},
        "recommended_role": recommended, "candidate_params": candidate, "baseline_params": baseline,
        "best_parameters_equal_predeclared_initial_search_mean": candidate == p["cem"]["initial_params"],
        "release_acceptance": release, "freeze_to_first_test_reset_seconds": (first_test - utc(freeze["frozen_at_utc"])).total_seconds(),
        "all_independent_motion_checks_passed": all(row["independent_motion_checks_passed"] for row in reports),
        "task_failures_and_inaccurate_cases": diagnostics, "all_episode_audits": reports,
        "engineering_spill_permission_did_not_relax_formal_task": True, "automatic_gui_deployment": False,
        "engineering_initialization_audit": engineering,
        "limitations": [
            "Aggregate model mass and fixed initial-label count partitions are independently recomputed from logs. Per-particle positions/velocities/mass arrays are not serialized, so their full-state finite/unchanged assertions remain the frozen collector's evidence, not a second liquid solve.",
            "240Hz checks cover q/dq/ddq; independent raw-URDF FK, TCP and analytic grasp comparisons use 60Hz endpoints. This does not establish torque, friction-grasp or collision safety for a real robot.",
            "Partial fluid failures can leave an unlogged in-progress state; their last available liquid time/basis and terminal motion evidence are retained instead of inventing exact-5s water truth.",
            "Independent motion checks apply to logged endpoints and retained dense failure evidence. For a failed post frame lacking its controller status, u is reconstructed from the preceding committed C1 command and explicitly counted; unavailable full particle state is not recreated.",
            "The engineering opt-in admitted an already-spilling initial controller only for search; release still requires zero spill, every-case five-second home/stability/late-drop success, MAE<=0.1 and worst<=0.2mL overall and per target.",
            "Volumes are privileged fixed-particle truth at 0.042875mL quantum, not calibrated camera/scale or hardware accuracy. Holdout results are not used to change parameters or validation selection."
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_run(args.run, root=args.root)
        output = (args.output or args.run / "fast_run_audit.json").resolve()
        if output.exists():
            prior = read(output)
            need(isinstance(prior, dict) and prior.get("audit_schema_version") == 1 and "audit_tool_sha256" in prior,
                 "Refusing to overwrite a non-audit artifact")
    except (AuditError, KeyError, ValueError, OSError, StopIteration, TypeError) as error:
        print("FAST AUDIT REFUSED: " + str(error), file=sys.stderr)
        return 2
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"artifact_integrity_verified": True, "episodes": 138,
                      "release": report["release_acceptance"], "output": str(output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
