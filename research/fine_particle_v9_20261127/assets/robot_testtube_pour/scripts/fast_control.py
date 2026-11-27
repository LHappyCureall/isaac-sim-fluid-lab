"""Five-physical-second pouring on a joint-time-parameterized existing path.

This controller changes only kinematic tube commands. Liquid remains live PBF.
The geometry's u coordinate is normalized joint-limit time, NOT elapsed task
time and NOT the old source-time coordinate. |du/dt| <= 1 and |d2u/dt2| <= 3
are controller limits; actual joint velocity/acceleration still need auditing.
"""

import bisect
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


POLICY_KIND = "joint_time_fast_pouring_v3"
DEFAULT_PARAMS = {
    "transport_rate_fraction": 1.0,
    "fast_tilt_deg_s": 140.0,
    "meter_tilt_deg_s": 40.0,
    "return_tilt_deg_s": 180.0,
    "tail_lead_s": 0.50,
    "stop_bias_ml": 2.0,
}
PARAMETER_NAMES = tuple(DEFAULT_PARAMS)
PARAMETER_UNITS = {
    "transport_rate_fraction": "fraction of joint-time u-rate limit",
    "fast_tilt_deg_s": "degrees per physical second",
    "meter_tilt_deg_s": "degrees per physical second",
    "return_tilt_deg_s": "degrees per physical second",
    "tail_lead_s": "physical seconds",
    "stop_bias_ml": "milliliters",
}
PARAMETER_BOUNDS = {
    "transport_rate_fraction": (0.60, 1.0),
    "fast_tilt_deg_s": (45.0, 140.0),
    "meter_tilt_deg_s": (10.0, 80.0),
    "return_tilt_deg_s": (70.0, 180.0),
    "tail_lead_s": (0.02, 0.80),
    "stop_bias_ml": (-5.0, 10.0),
}
DEADLINE_SECONDS = 5.0
RETURN_NO_LATER_SECONDS = 2.90
METER_REMAINING_ML = 8.0
STABLE_FRAMES = 12
JOINT_ACCELERATION_COMMAND_CAP_RAD_S2 = 24.5
COMMAND_CHECK_SAMPLES = 9


class FastPolicy:
    name = POLICY_KIND
    policy_kind = POLICY_KIND

    def __init__(self, params=None, **parameters):
        supplied = {} if params is None else dict(params)
        if set(supplied) & set(parameters):
            raise ValueError("A fast parameter was supplied twice.")
        supplied.update(parameters)
        if set(supplied) - set(DEFAULT_PARAMS):
            raise ValueError("Unknown fast-policy parameter.")
        self.params = {**DEFAULT_PARAMS, **supplied}
        for name, (low, high) in PARAMETER_BOUNDS.items():
            value = self.params[name]
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or not low <= float(value) <= high:
                raise ValueError(f"Fast parameter outside finite bounds: {name}")
            self.params[name] = float(value)

    def act(self, observation):
        values = {name: float(observation[name]) for name in
                  ("target_ml", "received_ml", "inflight_ml", "flow_ml_s")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Fast policy observations must be finite.")
        prediction = (values["received_ml"] + values["inflight_ml"]
                      + max(0.0, values["flow_ml_s"]) * self.params["tail_lead_s"])
        gap = values["target_ml"] - (values["received_ml"] + values["inflight_ml"])
        return {"stop": prediction + self.params["stop_bias_ml"] >= values["target_ml"],
                "predicted_final_ml": prediction, "remaining_committed_ml": gap,
                "requested_tilt_deg_s": self.params["meter_tilt_deg_s"] if gap <= METER_REMAINING_ML
                else self.params["fast_tilt_deg_s"], "stop_bias_ml": self.params["stop_bias_ml"]}

    def save(self, path, metadata=None):
        data = {"schema_version": 1, "policy_kind": POLICY_KIND, "parameters": dict(self.params),
                "observation_units": {"volume": "mL", "flow": "mL/s", "tilt": "degrees",
                                      "physical_time": "seconds"},
                "action_units": {"tilt_rates": "degrees/physical-second", "u_rate": "joint-time-u/physical-second"},
                "deadline_physical_seconds": DEADLINE_SECONDS,
                "metadata": {} if metadata is None else metadata}
        content = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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
        return data

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("policy_kind") != POLICY_KIND:
            raise ValueError("Unsupported fast pouring checkpoint.")
        if set(data.get("parameters", {})) != set(DEFAULT_PARAMS):
            raise ValueError("Fast checkpoint must contain all six parameters.")
        json.dumps(data, allow_nan=False)
        return cls(data["parameters"])


class TargetPourController:
    """Transport, bulk/meter, one predictive return, then stable full-home end.

    Uses the same C1 polynomial segment evaluator as V2, without endpoint snaps.
    There are no micro-pulse loops. A missed five-second deadline is latched;
    later diagnostics may keep returning home but can never become a success.
    """

    def __init__(self, geometry, target_ml=30.0, spacing=0.0035, policy=None):
        self.geometry = geometry
        self.spacing = float(spacing)
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive.")
        if policy is None:
            self.policy = FastPolicy()
        elif isinstance(policy, dict):
            self.policy = FastPolicy(policy)
        elif getattr(policy, "name", None) == POLICY_KIND:
            self.policy = FastPolicy(policy.params)
        else:
            raise ValueError("Expected a FastPolicy or its parameter dictionary.")
        self.particle_ml = self.spacing ** 3 * 1e6
        self.home_source = self.upright_source = 0.0
        self.pretilt_source = float(geometry.u_for_tilt(78.0))
        self.max_source = float(geometry.DURATION)
        self.max_speed, self.max_acceleration = 1.0, 3.0
        self._home_position, self._home_quaternion = geometry.tube_pose(0.0)
        self._home_joints = np.asarray(geometry.joint_positions(0.0), dtype=float)
        self.reset(target_ml)

    def reset(self, target_ml=None):
        if target_ml is not None:
            self.target_ml = float(target_ml)
        if not math.isfinite(self.target_ml) or not 0 < self.target_ml <= 45:
            raise ValueError("target_ml must be in (0,45].")
        self.phase, self.done = "prepare", False
        self.timed_out = False
        self._last_t = None
        self._last_points = None
        self._last_released = 0.0
        self._flow = 0.0
        self._last_received_count = None
        self._stable_frames = 0
        self._return_reason = None
        self._last_angular_velocity = 0.0
        self._segments = [(0.0, 0.0, 0.0, 0.0, 1 / 60, self.pretilt_source, ())]
        self._times = [0.0]
        self._last_command = {}
        self.status = {"phase": self.phase, "done": False, "target_ml": self.target_ml,
                       "policy_name": POLICY_KIND, "policy_params": dict(self.policy.params),
                       "deadline_s": DEADLINE_SECONDS, "timed_out": False}
        return dict(self.status)

    @staticmethod
    def _evaluate(segment, t):
        # The committed frame is the same C1 polynomial used by V2. Queries
        # beyond that frame use an explicitly bounded braking continuation.
        # Newton asks for a future pose when estimating the last substep twist;
        # constant-velocity extrapolation could otherwise leave the path before
        # the next feedback command has been committed.
        t0, source, velocity, acceleration, horizon, _goal = segment[:6]
        elapsed = max(0.0, float(t) - t0)
        h = min(elapsed, horizon)
        source += velocity * h + 0.5 * acceleration * h * h
        velocity += acceleration * h
        future = max(0.0, elapsed - horizon)
        if future > 0 and len(segment) > 6:
            continuation = segment[6]
            for start, u0, v0, a0, duration in continuation:
                if future < start + duration:
                    tau = max(0.0, future - start)
                    return u0 + v0 * tau + 0.5 * a0 * tau * tau, v0 + a0 * tau
                source = u0 + v0 * duration + 0.5 * a0 * duration * duration
                velocity = v0 + a0 * duration
            if continuation:
                return source, velocity
        # Six-field segments are accepted for old CPU fixtures only. Live
        # commands always carry a continuation, including a rest segment.
        source += velocity * future
        return source, velocity

    def source_time(self, t):
        index = max(0, bisect.bisect_right(self._times, float(t)) - 1)
        source = self._evaluate(self._segments[index], t)[0]
        if source < -1e-7 or source > self.max_source + 1e-7:
            raise RuntimeError("Fast commanded u escaped the validated joint-time path.")
        # Floating-point roundoff only, not a physical endpoint position snap.
        return float(min(self.max_source, max(0.0, source)))

    def tube_pose(self, t):
        return self.geometry.tube_pose(self.source_time(t))

    def _rate_cap(self, source, degrees_per_second, direction=1.0):
        # Limit speed before entering a region with a larger angle derivative,
        # rather than attempting an impossible instantaneous deceleration there.
        _, velocity = self._evaluate(self._segments[-1], self._last_t or 0.0)
        cap = self.policy.params["transport_rate_fraction"]
        lookahead = max(abs(velocity), cap) ** 2 / (2 * self.max_acceleration) + cap / 60
        end = float(np.clip(source + direction * lookahead, 0.0, self.max_source))
        derivative_deg = max(abs(math.degrees(float(self.geometry.angle_derivative(u))))
                             for u in np.linspace(source, end, 17))
        if derivative_deg > 1e-8:
            cap = min(cap, float(degrees_per_second) / derivative_deg)
        return float(min(self.max_speed, max(0.0, cap)))

    def _joint_acceleration_interval(self, source, velocity):
        low, high = self.geometry.joint_u_acceleration_bounds(
            float(source), float(velocity), JOINT_ACCELERATION_COMMAND_CAP_RAD_S2)
        return max(-self.max_acceleration, float(low)), min(self.max_acceleration, float(high))

    def _checked_acceleration(self, source, velocity, desired, duration, samples=COMMAND_CHECK_SAMPLES):
        """Project a proposed constant acceleration into sampled joint bounds.

        Both q_u*u_ddot and q_uu*u_dot**2 are included. Bounds are evaluated at
        the current state and within the proposed frame; clipping only at the
        first state is insufficient near a changing path derivative.
        """
        acceleration = float(np.clip(desired, -self.max_acceleration, self.max_acceleration))
        for _ in range(8):
            low, high = -self.max_acceleration, self.max_acceleration
            check_times = list(np.linspace(0.0, duration, samples))
            if acceleration and 0 < -velocity / acceleration < duration:
                check_times.append(-velocity / acceleration)
            for h in check_times:
                u = source + velocity * h + .5 * acceleration * h * h
                v = velocity + acceleration * h
                if not -1e-9 <= u <= self.max_source + 1e-9 or abs(v) > self.max_speed + 1e-8:
                    return None
                local_low, local_high = self._joint_acceleration_interval(
                    float(np.clip(u, 0.0, self.max_source)), v)
                low, high = max(low, local_low), min(high, local_high)
            if low > high + 1e-10:
                return None
            projected = float(np.clip(acceleration, low, high))
            if abs(projected - acceleration) <= 1e-9:
                return projected
            acceleration = projected
        return None

    def _braking_continuation(self, source, velocity, goal):
        """Construct a C1 joint-aware stop without clipping a position.

        This path is a safe answer to uncommitted future-pose queries, not an
        open-loop replacement for the next real 60 Hz feedback update.
        """
        pieces = []
        elapsed = 0.0
        u, v = float(source), float(velocity)
        direction = 1.0 if v >= 0 else -1.0
        toward_goal = (float(goal) - u) * v >= 0
        boundary = float(goal) if toward_goal else (self.max_source if v > 0 else 0.0)
        for _ in range(240):
            if abs(v) <= 1e-12:
                pieces.append((elapsed, u, 0.0, 0.0, 1.0))
                return tuple(pieces)
            duration = min(1 / 60, abs(v) / self.max_acceleration)
            acceleration = self._checked_acceleration(u, v, -direction * self.max_acceleration,
                                                      duration, samples=5)
            if acceleration is None or acceleration * direction >= -1e-9:
                return None
            duration = min(duration, -v / acceleration)
            next_u = u + v * duration + .5 * acceleration * duration * duration
            next_v = v + acceleration * duration
            if (next_u < -1e-9 or next_u > self.max_source + 1e-9
                    or direction * (next_u - boundary) > 1e-9):
                return None
            pieces.append((elapsed, u, v, acceleration, duration))
            elapsed += duration
            u, v = next_u, next_v
        return None

    def _command(self, t, goal, cap, dt):
        source, velocity = self._evaluate(self._segments[-1], t)
        distance = float(goal) - source
        direction = 1.0 if distance >= 0 else -1.0
        d = abs(distance)
        toward_velocity = direction * velocity
        # Discrete stopping envelope: after the next constant-acceleration
        # frame, retain enough distance to brake the resulting velocity to zero.
        discriminant = ((self.max_acceleration * dt) ** 2
                        - 4 * self.max_acceleration * dt * toward_velocity
                        + 8 * self.max_acceleration * d)
        braking_cap = max(0.0, 0.5 * (-self.max_acceleration * dt + math.sqrt(max(0.0, discriminant))))
        desired = direction * min(float(cap), braking_cap)
        desired_acceleration = float(np.clip((desired - velocity) / dt,
                                             -self.max_acceleration, self.max_acceleration))
        candidates = [desired_acceleration]
        # The nearest feasible proposal is preferred. Alternative accelerations
        # may be needed to retain a valid stop at a changing joint-path bound.
        candidates.extend(sorted(np.linspace(-self.max_acceleration, self.max_acceleration, 25),
                                 key=lambda value: abs(value - desired_acceleration)))
        selected = None
        checked = set()
        for proposed in candidates:
            horizon = dt
            acceleration = self._checked_acceleration(source, velocity, proposed, horizon)
            if acceleration is None or round(acceleration, 10) in checked:
                continue
            checked.add(round(acceleration, 10))
            next_velocity = velocity + acceleration * horizon
            if velocity * next_velocity < 0 and desired * velocity >= 0:
                # A within-frame stop is followed by rest, not a small reversal.
                horizon = -velocity / acceleration
                next_velocity = 0.0
            if abs(next_velocity) > max(abs(velocity), float(cap)) + 1e-9:
                continue
            next_source = source + velocity * horizon + .5 * acceleration * horizon * horizon
            if (not -1e-9 <= next_source <= self.max_source + 1e-9
                    or (toward_velocity > 0 and direction * (next_source - goal) > 1e-9)):
                continue
            continuation = self._braking_continuation(next_source, next_velocity, goal)
            if continuation is None:
                continue
            selected = (acceleration, horizon, next_source, next_velocity, continuation)
            break
        if selected is None:
            raise RuntimeError("No joint-safe bounded C1 command and braking continuation remain inside the fast path.")
        acceleration, horizon, next_source, next_velocity, continuation = selected
        segment = (float(t), source, velocity, acceleration, horizon, float(goal), continuation)
        if self._times[-1] == float(t):
            self._segments[-1] = segment
        else:
            self._times.append(float(t))
            self._segments.append(segment)
        self._last_command = {"command_goal_source": float(goal), "command_max_rate": float(cap),
            "command_acceleration": float(acceleration), "command_source_velocity": float(velocity),
            "command_next_source_velocity": float(next_velocity), "command_horizon_s": float(horizon),
            "command_start_u": float(source), "command_start_physical_s": float(t),
            "command_joint_acceleration_cap_rad_s2": JOINT_ACCELERATION_COMMAND_CAP_RAD_S2,
            "future_braking_duration_s": float(sum(piece[4] for piece in continuation[:-1])),
            "future_braking_stop_u": float(continuation[-1][1])}

    def update(self, t, positions, labels, velocities=None):
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Physical time must be finite and nonnegative.")
        if self._last_t is not None and t <= self._last_t:
            if abs(t - self._last_t) < 1e-9:
                return dict(self.status)
            raise ValueError("Physical time must increase; reset before restarting.")
        dt = t - self._last_t if self._last_t is not None else 1 / 60
        points = np.asarray(positions, dtype=np.float64)
        groups = np.asarray(labels)
        if points.ndim != 2 or points.shape[1] != 3 or groups.shape != (len(points),) or not np.isfinite(points).all():
            raise ValueError("positions must be finite Nx3 with matching fixed labels.")
        velocity = (np.asarray(velocities, dtype=float) if velocities is not None else
                    (points - self._last_points) / dt if self._last_points is not None else np.zeros_like(points))
        if velocity.shape != points.shape or not np.isfinite(velocity).all():
            raise ValueError("velocities must be finite and match positions.")
        g = self.geometry
        source = self.source_time(t)
        _, source_velocity = self._evaluate(self._segments[-1], t)
        previous_segment = self._segments[-1]
        actual_acceleration = previous_segment[3] if t - previous_segment[0] <= previous_segment[4] + 1e-9 else 0.0
        mouth, angle = g.mouth_pose(source)
        origin, quaternion = g.tube_pose(source)
        tube = g._cavity_mask((points - np.asarray(origin)) @ g.rotation_y(angle))
        center = np.asarray(g.BEAKER_CENTER)
        delta = points - center
        beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2)
                  & (delta[:, 2] >= 0) & (delta[:, 2] <= g.BEAKER_HEIGHT))
        initial_tube = groups == 1
        free = initial_tube & ~(tube | beaker)
        airborne = free & (points[:, 2] > g.TABLE_TOP_Z + self.spacing)
        rim_z = center[2] + g.BEAKER_HEIGHT
        discriminant = velocity[:, 2] ** 2 + 2 * 9.81 * np.maximum(points[:, 2] - rim_z, 0)
        fall_time = (velocity[:, 2] + np.sqrt(discriminant)) / 9.81
        impact_xy = points[:, :2] + velocity[:, :2] * fall_time[:, None]
        catch = np.sum((impact_xy - center[:2]) ** 2, axis=1) <= (g.BEAKER_RADIUS + self.spacing / 2) ** 2
        catch |= (points[:, 2] <= rim_z) & (np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2)
        received_count = int(np.count_nonzero(initial_tube & beaker))
        released_count = int(np.count_nonzero(initial_tube & ~tube))
        airborne_count = int(np.count_nonzero(airborne))
        received = received_count * self.particle_ml
        released = released_count * self.particle_ml
        inflight = float(np.count_nonzero(airborne & catch) * self.particle_ml)
        instantaneous_flow = max(0.0, (released - self._last_released) / dt)
        self._flow += (1 - math.exp(-dt / 0.06)) * (instantaneous_flow - self._flow)
        action = self.policy.act({"target_ml": self.target_ml, "received_ml": received,
                                  "inflight_ml": inflight, "flow_ml_s": self._flow})
        tilt = math.degrees(float(angle))
        pretilt_rest = bool(abs(source - self.pretilt_source) <= 1e-5 and abs(source_velocity) <= 1e-4)
        if self.phase == "prepare" and pretilt_rest:
            self.phase = "meter"
        if self.phase in ("prepare", "bulk", "meter") and action["stop"]:
            self.phase, self._return_reason = "return", "predicted_target_reached"
        if self.phase in ("prepare", "bulk", "meter") and t >= RETURN_NO_LATER_SECONDS - 1e-9:
            self.phase, self._return_reason = "return", "deadline_forced_return"
        joints = np.asarray(g.joint_positions(source), dtype=float)
        joint_velocity = np.asarray(g.joint_derivative(source), dtype=float) * source_velocity
        home_position_error = float(np.linalg.norm(np.asarray(origin) - self._home_position))
        home_orientation_error = float(2 * math.acos(np.clip(abs(np.dot(quaternion, self._home_quaternion)), 0, 1)))
        home_joint_error = float(np.max(np.abs(joints - self._home_joints)))
        home_static = bool(home_position_error <= .0005 and home_orientation_error <= math.radians(.1)
                           and home_joint_error <= .001 and abs(source_velocity) <= 1e-4
                           and np.max(np.abs(joint_velocity)) <= .001)
        if self.phase == "return" and home_static:
            self.phase = "settle"
        stable_now = bool(home_static and airborne_count == 0 and received_count == self._last_received_count)
        if self.phase == "settle" and stable_now:
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        if not self.timed_out and self.phase == "settle" and self._stable_frames >= STABLE_FRAMES and t <= DEADLINE_SECONDS + 1e-9:
            self.done, self.phase = True, "done"
        if not self.done and t >= DEADLINE_SECONDS - 1e-9:
            self.timed_out, self.phase = True, "timeout_return"
        if self.timed_out:
            self.done = False  # Diagnostic post-steps can never upgrade a timeout.
        returning = self.phase in ("return", "settle", "done", "timeout_return")
        requested_degrees = (self.policy.params["return_tilt_deg_s"] if returning else
                             self.policy.params["fast_tilt_deg_s"] if self.phase == "prepare" else
                             self.policy.params["meter_tilt_deg_s"])
        goal = 0.0 if returning else self.pretilt_source if self.phase == "prepare" else self.max_source
        cap = self._rate_cap(source, requested_degrees, 1.0 if goal >= source else -1.0)
        self._command(t, goal, cap, dt)
        angular_velocity = math.degrees(float(g.angle_derivative(source))) * source_velocity
        angular_acceleration = (angular_velocity - self._last_angular_velocity) / dt
        tcp = np.asarray(origin) + g.rotation_y(angle) @ np.asarray(
            (0.0, 0.0, g.TUBE_LENGTH - g.TCP_BELOW_MOUTH))
        action.update(rate=cap, decision=self.phase, requested_tilt_deg_s=requested_degrees,
                      commanded_goal_u=goal, commanded_u_rate_cap=cap, return_reason=self._return_reason)
        self.status = {"phase": self.phase, "fast_phase": self.phase, "done": self.done,
            "target_ml": self.target_ml, "received_ml": received, "released_ml": released,
            "inflight_ml": inflight, "flow_ml_s": float(self._flow), "instantaneous_flow_ml_s": instantaneous_flow,
            "predicted_final_ml": action["predicted_final_ml"], "source_time": source,
            "tilt_deg": tilt, "error_ml": received - self.target_ml, "particle_count": len(points),
            "tube_remaining_ml": float(np.count_nonzero(initial_tube & tube) * self.particle_ml),
            "spilled_ml": float(np.count_nonzero(free & ~airborne) * self.particle_ml),
            "source_airborne_particles": airborne_count, "trim_count": 0, "pulse_count": 0,
            "policy_name": POLICY_KIND, "policy_params": dict(self.policy.params), "policy_action": action,
            "deadline_s": DEADLINE_SECONDS, "timed_out": self.timed_out,
            "completed_by_deadline": self.done and not self.timed_out,
            "return_reason": self._return_reason, "return_no_later_s": RETURN_NO_LATER_SECONDS,
            "pretilt_rest_complete": pretilt_rest, "post_pretilt_meter_rate_only": True,
            "stable_frames": self._stable_frames, "required_stable_frames": STABLE_FRAMES,
            "home_position_error_m": home_position_error, "home_orientation_error_rad": home_orientation_error,
            "home_joint_error_rad": home_joint_error, "home_static": home_static,
            "actual_u_velocity": float(source_velocity), "actual_u_acceleration": float(actual_acceleration),
            "u": source, "u_units": "normalized_joint_limit_time_coordinate",
            "u_rate_units": "u/physical-second", "u_acceleration_units": "u/physical-second^2",
            "old_source_time_s": float(g.source_from_u(source)),
            "angular_velocity_deg_s": float(angular_velocity), "angular_acceleration_deg_s2": float(angular_acceleration),
            "tcp_position_m": tcp.tolist(), "joint_positions_rad": joints.tolist(),
            "joint_velocity_rad_s": joint_velocity.tolist(),
            "fast_dose_within_0_1_ml": bool(self.done and abs(received - self.target_ml) <= .1),
            **self._last_command}
        self._last_t, self._last_points = t, points.copy()
        self._last_received_count, self._last_released = received_count, released
        self._last_angular_velocity = angular_velocity
        return dict(self.status)
