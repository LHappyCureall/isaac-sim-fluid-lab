"""Independent upright-lift V2 planned-return controller; no liquid-state writes.

Derived from frozen direct controller 35adf6fc22c936c8586299c05614e996b9f6575e87487e2286a321d516ef3fff.
V2 adds an exact reverse-prepare clock after the 78-degree quiet hold and a
pure joint-safe retract rollout for deadline budgeting. The unchanged 5-second
hard deadline remains authoritative; the prediction margin is heuristic.

Preparation runs the already C2 joint clock directly. Metering and return use
C1 committed path segments with sampled physical joint limits. Clock-rate
changes at the 0/78/125 anchors are physically C2 because q' and q'' vanish.
Beyond a committed frame, pose queries brake safely instead of extrapolating
an unobserved pouring command forever. Every query is a pure function.
"""

import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


POLICY_KIND = "upright_lift_fixed_mouth_planned_return_v2"
DEFAULT_PARAMS = {"prepare_fraction": .9, "dose_tilt_deg_s": 40.0,
                  "return_fraction": 1.0, "tail_lead_s": .25,
                  "stop_bias_ml": .2, "max_dose_u_accel": 6.0}
PARAMETER_NAMES = tuple(DEFAULT_PARAMS)
PARAMETER_BOUNDS = {"prepare_fraction": (.8, 1.0), "dose_tilt_deg_s": (20.0, 120.0),
                    "return_fraction": (.8, 1.0), "tail_lead_s": (0.0, .80),
                    "stop_bias_ml": (-5.0, 10.0), "max_dose_u_accel": (2.0, 10.0)}
BOUNDS = PARAMETER_BOUNDS
EXPECTED_GEOMETRY_SHA256 = "6f5ee0f694958ede3f111b1d27134c6571c4d7ef0ce5dae96425c4829638bfa9"
EXPECTED_CACHE_SHA256 = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
DEADLINE_SECONDS = 5.0
HOME_QUIET_SECONDS = .2
STABLE_FRAMES = 12
JOINT_ACCELERATION_CAP_RAD_S2 = 25.0
PLANNING_JOINT_ACCELERATION_CAP_RAD_S2 = 24.0
SAFE_FUTURE_BRAKE = 1.5
ANCHOR_HOLD_MAX_SECONDS = .5
ANCHOR_HOLD_QUIET_SECONDS = .1
RETURN_PLANNER_STEP_SECONDS = 1 / 60
RETURN_SCHEDULING_MARGIN_SECONDS = 2 / 60
PARENT_CONTROL_SHA256 = "2aac9bedab9b72619a4166e86f161c92991e5a9ea47c47930e9f2cd20548675d"


class DirectPolicy:
    name = POLICY_KIND

    def __init__(self, params=None, **parameters):
        values = {} if params is None else dict(params)
        if set(values) & set(parameters) or (set(values) | set(parameters)) - set(DEFAULT_PARAMS):
            raise ValueError("Unknown or duplicate direct-policy parameter.")
        values = {**DEFAULT_PARAMS, **values, **parameters}
        for name, (low, high) in PARAMETER_BOUNDS.items():
            value = values[name]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value)) or not low <= float(value) <= high):
                raise ValueError("Direct-policy parameter outside finite bounds: " + name)
        self.params = {name: float(values[name]) for name in PARAMETER_NAMES}

    def act(self, observation):
        target, received, inflight, flow = (float(observation[name]) for name in
                                           ("target_ml", "received_ml", "inflight_ml", "flow_ml_s"))
        if not all(math.isfinite(value) for value in (target, received, inflight, flow)) or not 0 < target <= 45:
            raise ValueError("Finite direct-policy volume observations are required.")
        predicted = received + inflight + max(0.0, flow) * self.params["tail_lead_s"]
        return {"stop": bool(predicted + self.params["stop_bias_ml"] >= target),
                "predicted_final_ml": predicted, "remaining_committed_ml": target - received - inflight,
                "requested_tilt_deg_s": self.params["dose_tilt_deg_s"],
                "prediction_basis": "received_plus_ballistic_catchable_inflight_plus_flow_tail"}

    def save(self, path, metadata=None):
        value = {"schema_version": 1, "policy_kind": POLICY_KIND, "parameters": dict(self.params),
                 "observation_units": {"volume": "mL", "flow": "mL/s", "tilt_rate": "degrees/physical-second"},
                 "metadata": {} if metadata is None else metadata}
        content = json.dumps(value, indent=2, allow_nan=False) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             delete=False, suffix=".tmp") as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return value

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
        if (value.get("schema_version") != 1 or value.get("policy_kind") != POLICY_KIND
                or set(value.get("parameters", {})) != set(PARAMETER_NAMES)):
            raise ValueError("Unsupported direct-policy checkpoint; old fast/V2 policies are not interchangeable.")
        return cls(value["parameters"])


FastPourPolicy = DirectPolicy


class TargetPourController:
    max_speed = 1.0
    joint_acceleration_cap = JOINT_ACCELERATION_CAP_RAD_S2

    def __init__(self, geometry, target_ml=30.0, spacing=.0035, policy=None):
        self.geometry = geometry
        if hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest() != EXPECTED_GEOMETRY_SHA256:
            raise ValueError("Direct geometry differs from the independently checked prototype.")
        if getattr(geometry, "CACHE_SHA256", None) != EXPECTED_CACHE_SHA256:
            raise ValueError("Direct geometry reference cache differs from its pinned SHA.")
        if policy is None:
            policy = DirectPolicy()
        elif isinstance(policy, dict):
            policy = DirectPolicy(policy)
        elif getattr(policy, "name", None) == POLICY_KIND:
            policy = DirectPolicy(policy.params)
        else:
            raise ValueError("The direct controller requires its own DirectPolicy.")
        self.policy, self.spacing = policy, float(spacing)
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("Particle spacing must be finite and positive.")
        self.particle_ml = self.spacing ** 3 * 1e6
        self.max_source = float(geometry.U_MAX)
        self.max_acceleration = self.policy.params["max_dose_u_accel"]
        self.reset(target_ml)

    def reset(self, target_ml=None):
        if target_ml is not None:
            self.target_ml = float(target_ml)
        if not math.isfinite(self.target_ml) or not 0 < self.target_ml <= 45:
            raise ValueError("target_ml must be in (0,45].")
        self.phase, self.done, self.timed_out = "prepare", False, False
        self._prepare_end = self.geometry.PREPARE_END_U / self.policy.params["prepare_fraction"]
        self._segments = [(0.0, 0.0, self.policy.params["prepare_fraction"], 0.0,
                           self._prepare_end, self.geometry.PREPARE_END_U, "prepare_clock")]
        self._times = [0.0]
        self._last_t = None
        self._last_points = None
        self._last_released = 0.0
        self._last_received_count = 0
        self._last_released_count = 0
        self._flow = 0.0
        self._stable_since = None
        self._stable_frames = 0
        self._return_reason = None
        self._completed_at = None
        self._hold_started = None
        self._hold_finished = None
        self._hold_stable_since = None
        self._hold_release_reason = None
        self._mode_event = "prepare_clock_started"
        self._last_command = {}
        self._home_clock_started = None
        self._return_budget = {}
        self._observed_tube_pose = None
        self._home_q = self.geometry.joint_positions(0)
        self._home_origin, self._home_quaternion = self.geometry.tube_pose(0)
        self.status = {"phase": self.phase, "done": False, "target_ml": self.target_ml,
                       "policy_name": POLICY_KIND, "policy_params": dict(self.policy.params)}
        return dict(self.status)

    def set_observed_tube_pose(self, pose_xyzw):
        """Use a copied actual collider pose for the next observation only.

        This never writes the collider or changes any planned/future pose.
        The caller supplies [x,y,z,qx,qy,qz,qw] immediately before update.
        """
        pose = np.asarray(pose_xyzw, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError("Observed tube collider pose must be finite XYZ+quaternion XYZW.")
        pose = pose.copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        self._observed_tube_pose = pose

    def _evaluate(self, segment, t):
        t0, u0, v0, acceleration, horizon, goal, kind = segment
        anchor_cut = kind == "C1_retract_to_78_commit"
        elapsed = max(0.0, float(t) - t0)
        if kind in ("prepare_clock", "reverse_prepare_clock"):
            if elapsed >= horizon:
                return float(goal), 0.0, 0.0, ("C2_home_anchor_hold" if kind == "reverse_prepare_clock"
                                                            else "C2_prepare_anchor_hold")
            return u0 + v0 * elapsed, v0, 0.0, ("C2_reverse_prepare_clock" if kind == "reverse_prepare_clock"
                                                          else "C2_prepare_clock")
        h = min(elapsed, horizon)
        source = u0 + v0 * h + .5 * acceleration * h * h
        velocity = v0 + acceleration * h
        if elapsed > horizon:
            extra = min(elapsed - horizon, abs(velocity) / SAFE_FUTURE_BRAKE)
            brake = -math.copysign(SAFE_FUTURE_BRAKE, velocity) if velocity else 0.0
            source += velocity * extra + .5 * brake * extra * extra
            velocity += brake * extra
            acceleration = brake if extra < abs(v0 + segment[3] * horizon) / SAFE_FUTURE_BRAKE else 0.0
            kind = "bounded_joint_safe_future_brake"
        if anchor_cut and source <= goal:
            return float(goal), 0.0, 0.0, "C2_78_cut_anchor_hold"
        if source <= 0:
            return 0.0, 0.0, 0.0, "C2_home_anchor_hold"
        if source >= self.max_source:
            return self.max_source, 0.0, 0.0, "C2_125_anchor_hold"
        return float(source), float(velocity), float(acceleration), kind

    def motion_state(self, t):
        index = max(0, bisect.bisect_right(self._times, float(t)) - 1)
        return self._evaluate(self._segments[index], t)

    def source_time(self, t):
        return self.motion_state(t)[0]

    def tube_pose(self, t):
        return self.geometry.tube_pose(self.source_time(t))

    def kinematics(self, t):
        u, velocity, acceleration, continuation = self.motion_state(t)
        first, second = self.geometry.joint_derivative(u), self.geometry.joint_second_derivative(u)
        return {"physical_time_s": float(t), "u": u, "u_rate": velocity, "u_acceleration": acceleration,
                "joint_position_rad": self.geometry.joint_positions(u).tolist(),
                "joint_velocity_rad_s": (first * velocity).tolist(),
                "joint_acceleration_rad_s2": (second * velocity * velocity + first * acceleration).tolist(),
                "continuation": continuation,
                "future_of_last_observation": self._last_t is None or float(t) > self._last_t}

    def pose_state(self, t):
        """Pure full-R finite-difference twist plus exact joint derivatives."""
        result = self.kinematics(t)
        low, high = max(0.0, float(t) - 1e-5), float(t) + 1e-5
        u_low, u_high = self.source_time(low), self.source_time(high)
        p_low, _ = self.geometry.tube_pose(u_low)
        p_high, _ = self.geometry.tube_pose(u_high)
        relative = self.geometry.tube_rotation(u_high) @ self.geometry.tube_rotation(u_low).T
        angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        skew = np.array((relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                         relative[1, 0] - relative[0, 1]))
        scale = .5 if angle < 1e-7 else angle / (2 * math.sin(angle))
        result.update(tube_origin_velocity_m_s=((p_high - p_low) / (high - low)).tolist(),
                      tube_angular_velocity_world_rad_s=(skew * scale / (high - low)).tolist())
        return result

    def _motion_safe(self, u, velocity, acceleration, horizon):
        limits = self.geometry.JOINT_VELOCITY_LIMITS_RAD_S
        peak_ratio = peak_acceleration = 0.0
        for elapsed in np.linspace(0.0, horizon, 9):
            source = u + velocity * elapsed + .5 * acceleration * elapsed ** 2
            rate = velocity + acceleration * elapsed
            if not -1.0000001 <= rate <= 1.0000001:
                return None
            if source <= 0 or source >= self.max_source:
                # Only physical C2 anchors may terminate the nominal clock.
                continue
            first, second = self.geometry.joint_derivative(source), self.geometry.joint_second_derivative(source)
            ratio = float(np.max(np.abs(first * rate) / limits))
            physical_acceleration = float(np.max(np.abs(second * rate * rate + first * acceleration)))
            peak_ratio, peak_acceleration = max(peak_ratio, ratio), max(peak_acceleration, physical_acceleration)
            if ratio > .999 or physical_acceleration > PLANNING_JOINT_ACCELERATION_CAP_RAD_S2:
                return None
        return peak_ratio, peak_acceleration

    def _select_acceleration(self, u, velocity, desired_rate, dt):
        requested = float(np.clip((desired_rate - velocity) / dt, -self.max_acceleration, self.max_acceleration))
        candidates = np.unique(np.r_[requested, 0., -SAFE_FUTURE_BRAKE, SAFE_FUTURE_BRAKE,
                                     np.linspace(-self.max_acceleration, self.max_acceleration, 33)])
        candidates = sorted(candidates, key=lambda value: (abs(value - requested), abs(value)))
        selected = None
        for acceleration in candidates:
            if abs(acceleration) > self.max_acceleration + 1e-9:
                continue
            evidence = self._motion_safe(u, velocity, float(acceleration), dt)
            if evidence is not None:
                selected = float(acceleration), evidence
                break
        if selected is None:
            raise RuntimeError("No joint-safe direct path command exists for the current state.")
        return selected

    def _retract_duration_estimate(self, u, velocity):
        """Pure rollout of the same 60-Hz joint-safe commands back to 78.

        This predicts motion time only. It does not predict liquid clearing
        time and does not mutate committed motion, observations or particles.
        """
        anchor, dt = self.geometry.PREPARE_END_U, RETURN_PLANNER_STEP_SECONDS
        if u <= anchor + 1e-12 and abs(velocity) < 1e-9:
            return 0.0
        for step in range(1, 301):
            acceleration, _ = self._select_acceleration(
                u, velocity, -self.policy.params["return_fraction"], dt)
            segment = (0., u, velocity, acceleration, dt, anchor, "C1_retract_to_78_commit")
            u, velocity, _, _ = self._evaluate(segment, dt)
            if u <= anchor + 1e-12 and abs(velocity) < 1e-9:
                return step * dt
        raise RuntimeError("Joint-safe retract planner did not reach the 78-degree anchor within 5s.")

    def _estimate_return_budget(self, u, velocity):
        retract = self._retract_duration_estimate(u, velocity)
        home_clock = self.geometry.PREPARE_END_U / self.policy.params["return_fraction"]
        total = (retract + ANCHOR_HOLD_MAX_SECONDS + home_clock + HOME_QUIET_SECONDS
                 + RETURN_SCHEDULING_MARGIN_SECONDS)
        return {"retract_motion_estimate_s": retract, "exact_reverse_prepare_clock_s": home_clock,
                "anchor_hold_reserved_s": ANCHOR_HOLD_MAX_SECONDS,
                "home_quiet_reserved_s": HOME_QUIET_SECONDS,
                "scheduling_margin_s": RETURN_SCHEDULING_MARGIN_SECONDS,
                "total_return_reserved_s": total,
                "return_deadline_physical_s": DEADLINE_SECONDS - total,
                "basis": "same_joint_safe_60Hz_retract_rollout_plus_exact_reverse_prepare_and_maximum_hold",
                "is_guaranteed_liquid_deadline": False}

    def _start_reverse_prepare_clock(self, t, u):
        anchor = self.geometry.PREPARE_END_U
        if abs(u - anchor) > 1e-9:
            raise RuntimeError("Reverse prepare clock must start at the exact physical C2 78-degree anchor.")
        if max(np.max(np.abs(self.geometry.joint_derivative(anchor))),
               np.max(np.abs(self.geometry.joint_second_derivative(anchor)))) > 1e-9:
            raise RuntimeError("Reverse prepare clock requires zero joint first/second derivatives at its anchor.")
        rate = -self.policy.params["return_fraction"]
        segment = (float(t), float(anchor), rate, 0., anchor / abs(rate), 0., "reverse_prepare_clock")
        if self._times[-1] == float(t):
            self._segments[-1] = segment
        else:
            self._segments.append(segment)
            self._times.append(float(t))
        self._home_clock_started = float(t)

    def _command(self, t, u, velocity, desired_rate, dt):
        acceleration, evidence = self._select_acceleration(u, velocity, desired_rate, dt)
        goal = (self.geometry.PREPARE_END_U if self.phase in ("retract_78", "hold_78") else
                0.0 if self.phase in ("home_return", "settle", "done", "timeout_return") else self.max_source)
        segment = (float(t), u, velocity, acceleration, dt, goal,
                   "C1_retract_to_78_commit" if self.phase == "retract_78" else "C1_joint_limited_commit")
        if self._times[-1] == float(t):
            self._segments[-1] = segment
        else:
            self._segments.append(segment)
            self._times.append(float(t))
        self._last_command = {"command_goal_source": segment[5], "command_max_rate": abs(desired_rate),
            "command_acceleration": acceleration, "command_source_velocity": velocity,
            "command_next_source_velocity": velocity + acceleration * dt,
            "command_horizon_s": dt, "planned_joint_velocity_limit_ratio": evidence[0],
            "planned_joint_acceleration_rad_s2": evidence[1], "motion_continuity": "C1; C2 at nominal clock anchors"}

    def update(self, t, positions, labels, velocities=None):
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Physical time must be finite and nonnegative.")
        if self._last_t is not None and t <= self._last_t:
            if abs(t - self._last_t) < 1e-9:
                return dict(self.status)
            raise ValueError("Reset before reversing physical time.")
        points, groups = np.asarray(positions, dtype=float), np.asarray(labels)
        if (points.ndim != 2 or points.shape[1] != 3 or groups.shape != (len(points),)
                or not np.isfinite(points).all() or np.any((groups != 0) & (groups != 1))):
            raise ValueError("Finite Nx3 particles and fixed 0/1 labels are required.")
        dt = t - self._last_t if self._last_t is not None else 1 / 60
        if velocities is None:
            velocity_points = (points - self._last_points) / dt if self._last_points is not None else np.zeros_like(points)
        else:
            velocity_points = np.asarray(velocities, dtype=float)
            if velocity_points.shape != points.shape or not np.isfinite(velocity_points).all():
                raise ValueError("Finite matching particle velocities are required.")
        u, source_velocity, source_acceleration, continuation = self.motion_state(t)
        g = self.geometry
        origin, quaternion = g.tube_pose(u)
        rotation = g.tube_rotation(u)
        observed = self._observed_tube_pose
        self._observed_tube_pose = None
        if observed is not None:
            origin, quaternion = observed[:3], observed[3:]
            x, y, z, w = quaternion
            rotation = np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                                 (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                                 (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
        tube = g._cavity_mask((points - origin) @ rotation)
        center = np.asarray(g.BEAKER_CENTER)
        delta = points - center
        beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2)
                  & (delta[:, 2] >= 0) & (delta[:, 2] <= g.BEAKER_HEIGHT))
        source = groups == 1
        free = source & ~(tube | beaker)
        airborne = free & (points[:, 2] > g.TABLE_TOP_Z + self.spacing)
        rim_z = center[2] + g.BEAKER_HEIGHT
        fall = (velocity_points[:, 2] + np.sqrt(velocity_points[:, 2] ** 2
                + 2 * 9.81 * np.maximum(points[:, 2] - rim_z, 0))) / 9.81
        impact = points[:, :2] + velocity_points[:, :2] * fall[:, None]
        catch = np.sum((impact - center[:2]) ** 2, axis=1) <= (g.BEAKER_RADIUS + .5 * self.spacing) ** 2
        catch |= (points[:, 2] <= rim_z) & (np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2)
        received_count = int(np.count_nonzero(source & beaker))
        released_count = int(np.count_nonzero(source & ~tube))
        received, released = received_count * self.particle_ml, released_count * self.particle_ml
        inflight, airborne_count = np.count_nonzero(airborne & catch) * self.particle_ml, int(np.count_nonzero(airborne))
        instantaneous_flow = max(0.0, (released - self._last_released) / dt)
        self._flow += (1 - math.exp(-dt / .06)) * (instantaneous_flow - self._flow)
        action = self.policy.act({"target_ml": self.target_ml, "received_ml": received,
                                  "inflight_ml": inflight, "flow_ml_s": self._flow})
        if self.phase == "prepare" and t >= self._prepare_end - 1e-9:
            self.phase = "dose"
            self._mode_event = "prepare_78_anchor_reached"
        # Compatibility with CPU/user aborts that request the old return name.
        if self.phase == "return":
            self.phase = "retract_78" if u >= g.PREPARE_END_U else "home_return"
        if self.phase == "dose":
            self._return_budget = self._estimate_return_budget(u, source_velocity)
        return_no_later = self._return_budget.get("return_deadline_physical_s", DEADLINE_SECONDS)
        if self.phase == "dose" and (action["stop"] or t >= return_no_later):
            self.phase = "retract_78"
            self._return_reason = "predicted_target_reached" if action["stop"] else "deadline_forced_return"
            self._mode_event = "cut_flow_retract_to_78"
        if self.phase == "retract_78" and u <= g.PREPARE_END_U + 1e-12 and abs(source_velocity) <= 1e-9:
            self.phase = "hold_78"
            self._hold_started = t
            self._hold_stable_since = None
            self._mode_event = "78_anchor_hold_started"
        if self.phase == "hold_78":
            anchor_quiet = (airborne_count == 0 and received_count == self._last_received_count
                            and released_count == self._last_released_count)
            if anchor_quiet:
                if self._hold_stable_since is None:
                    self._hold_stable_since = t
            else:
                self._hold_stable_since = None
            stable_hold = (self._hold_stable_since is not None
                           and t - self._hold_stable_since >= ANCHOR_HOLD_QUIET_SECONDS - 1e-9)
            if stable_hold or t - self._hold_started >= ANCHOR_HOLD_MAX_SECONDS - 1e-9:
                self._hold_release_reason = "source_and_received_stable_no_airborne" if stable_hold else "hold_maximum_reached"
                self._hold_finished = t
                self.phase = "home_return"
                self._mode_event = "leave_78_hold_" + self._hold_release_reason
        joints = g.joint_positions(u)
        joint_velocity = g.joint_derivative(u) * source_velocity
        home_position_error = float(np.linalg.norm(origin - self._home_origin))
        home_orientation_error = 2 * math.acos(float(np.clip(abs(np.dot(quaternion, self._home_quaternion)), 0, 1)))
        home_joint_error = float(np.max(np.abs(joints - self._home_q)))
        home_static = bool(u <= 1e-9 and home_position_error <= 1e-6 and home_joint_error <= 1e-8
                           and np.max(np.abs(joint_velocity)) <= 1e-6)
        if self.phase == "home_return" and home_static:
            self.phase = "settle"
            self._mode_event = "home_anchor_reached"
        quiet = home_static and airborne_count == 0 and received_count == self._last_received_count
        if self.phase in ("settle", "done") and quiet:
            if self._stable_since is None:
                self._stable_since = t
            self._stable_frames += 1
        else:
            self._stable_since, self._stable_frames = None, 0
        if (self.phase == "settle" and not self.timed_out and self._stable_frames >= STABLE_FRAMES
                and t - self._stable_since >= HOME_QUIET_SECONDS - 1e-9 and t <= DEADLINE_SECONDS + 1e-9):
            self.phase, self.done = "done", True
            self._completed_at = t
            self._mode_event = "home_quiet_complete"
        if not self.done and t >= DEADLINE_SECONDS - 1e-9:
            if self.phase == "hold_78" and self._hold_finished is None:
                self._hold_finished = t
                self._hold_release_reason = "deadline_timeout_interrupted_hold"
            self.phase, self.timed_out = "timeout_return", True
            self._mode_event = "physical_5s_timeout_latched"
        if self.timed_out:
            self.done = False
        exact_home_clock = (self.phase == "home_return" and
                            (self._home_clock_started is not None or
                             (abs(u - g.PREPARE_END_U) <= 1e-9 and abs(source_velocity) < 1e-9)))
        if exact_home_clock:
            if self._home_clock_started is None:
                self._start_reverse_prepare_clock(t, u)
            rate = self.policy.params["return_fraction"]
            self._last_command = {"command_goal_source": 0., "command_max_rate": rate,
                "command_acceleration": 0., "command_source_velocity": -rate,
                "command_next_source_velocity": -rate,
                "command_horizon_s": max(0., self._home_clock_started + g.PREPARE_END_U / rate - t),
                "motion_continuity": "C2 physical reverse prepare clock; coordinate-rate switch only at zero q'/q'' anchor"}
        elif self.phase != "prepare":
            returning = self.phase in ("retract_78", "home_return", "settle", "done", "timeout_return")
            angular_derivative = abs(math.degrees(g.angle_derivative(u)))
            rate = (self.policy.params["return_fraction"] if returning else
                    min(1.0, self.policy.params["dose_tilt_deg_s"] / max(angular_derivative, 1e-9)))
            if returning and home_static:
                rate = 0.0
            if self.phase == "hold_78":
                rate = 0.0
            self._command(t, u, source_velocity, -rate if returning else rate, dt)
        else:
            rate = self.policy.params["prepare_fraction"]
            self._last_command = {"command_goal_source": g.PREPARE_END_U, "command_max_rate": rate,
                "command_acceleration": 0.0, "command_source_velocity": source_velocity,
                "command_next_source_velocity": source_velocity, "command_horizon_s": max(0., self._prepare_end - t),
                "motion_continuity": "C2 physical prepare clock; q and qdot/qddot vanish at its phase anchors"}
        action.update(rate=float(rate), decision=self.phase, return_reason=self._return_reason)
        self.status = {"phase": self.phase, "direct_phase": self.phase, "done": self.done,
            "target_ml": self.target_ml, "received_ml": float(received), "released_ml": float(released),
            "inflight_ml": float(inflight), "flow_ml_s": float(self._flow), "instantaneous_flow_ml_s": instantaneous_flow,
            "predicted_final_ml": action["predicted_final_ml"], "source_time": u, "u": u,
            "tilt_deg": math.degrees(g.mouth_pose(u)[1]), "error_ml": float(received - self.target_ml),
            "particle_count": len(points), "tube_remaining_ml": float(np.count_nonzero(source & tube) * self.particle_ml),
            "spilled_ml": float(np.count_nonzero(free & ~airborne) * self.particle_ml),
            "source_airborne_particles": airborne_count, "policy_name": POLICY_KIND,
            "policy_params": dict(self.policy.params), "policy_action": action,
            "timed_out": self.timed_out, "completed_by_deadline": self.done and not self.timed_out,
            "completed_physical_time_s": self._completed_at,
            "home_static": home_static, "home_position_error_m": home_position_error,
            "home_orientation_error_rad": home_orientation_error, "home_joint_error_rad": home_joint_error,
            "stable_frames": self._stable_frames, "required_stable_frames": STABLE_FRAMES,
            "home_quiet_seconds": 0.0 if self._stable_since is None else t - self._stable_since,
            "actual_u_velocity": source_velocity, "actual_u_acceleration": source_acceleration,
            "joint_positions_rad": joints.tolist(), "joint_velocity_rad_s": joint_velocity.tolist(),
            "tcp_position_m": (origin + rotation @ np.array([0., 0., g.TUBE_LENGTH - g.TCP_BELOW_MOUTH])).tolist(),
            "full_R_containment": True, "prepare_end_physical_s": self._prepare_end,
            "tube_pose_measurement_basis": "observed_collider_float_pose" if observed is not None else "planned_double_pose",
            "return_no_later_s": return_no_later, "return_reason": self._return_reason,
            "return_budget": dict(self._return_budget),
            "home_return_clock_start_physical_s": self._home_clock_started,
            "home_return_uses_exact_reverse_prepare": self._home_clock_started is not None,
            "mode_event": self._mode_event, "anchor_hold_start_physical_s": self._hold_started,
            "anchor_hold_elapsed_s": 0.0 if self._hold_started is None else
                (t if self._hold_finished is None else self._hold_finished) - self._hold_started,
            "anchor_hold_stable_seconds": 0.0 if self._hold_stable_since is None else
                (t if self._hold_finished is None else self._hold_finished) - self._hold_stable_since,
            "anchor_hold_max_seconds": ANCHOR_HOLD_MAX_SECONDS,
            "anchor_hold_required_quiet_seconds": ANCHOR_HOLD_QUIET_SECONDS,
            "anchor_hold_release_reason": self._hold_release_reason,
            "future_continuation": continuation, "correction_count": 0, "correction_enabled": False,
            "deadline_s": DEADLINE_SECONDS, **self._last_command}
        self._last_t, self._last_points = t, points.copy()
        self._last_received_count, self._last_released = received_count, float(released)
        self._last_released_count = released_count
        return dict(self.status)
