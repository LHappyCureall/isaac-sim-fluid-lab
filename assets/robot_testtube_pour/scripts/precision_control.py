"""Experimental closed-loop particle-pulse dosing; no liquid-state mutations.

The inherited bounded source-time command is reused with early braking. The
V2 segment evaluator removes the old endpoint position/velocity snap. This
controller adds local retraction, settling and repeated fine probes. Its minimum
controllable dose must be measured with live PBF; a
single released particle is not a promise that the final pulse is one particle.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


_BASE_PATH = Path(__file__).resolve().with_name("target_control.py")
_BASE_SPEC = importlib.util.spec_from_file_location("pour_precision_original_control", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

POLICY_KIND = "closed_loop_particle_pulse_precision_v2"
DEFAULT_PARAMS = {
    "reserve_ml": 2.5,
    "probe_rate": 0.003,
    "pulse_particles": 2,
    "retract_source": 0.04,
    "fine_retract_source": 0.008,
}
PARAMETER_NAMES = tuple(DEFAULT_PARAMS)
PARAMETER_BOUNDS = {
    "reserve_ml": (1.5, 4.0),
    "probe_rate": (0.001, 0.010),
    "pulse_particles": (1, 3),
    "retract_source": (0.02, 0.08),
    "fine_retract_source": (0.004, 0.02),
}


class PrecisionPolicy:
    """Bounded controller configuration, distinct from the old CEM stop model."""

    name = POLICY_KIND
    policy_kind = POLICY_KIND

    def __init__(self, params=None, **parameters):
        supplied = {} if params is None else dict(params)
        if set(supplied) & set(parameters):
            raise ValueError("A precision parameter was supplied twice.")
        supplied.update(parameters)
        if set(supplied) - set(DEFAULT_PARAMS):
            raise ValueError("Unknown precision-policy parameter.")
        values = {**DEFAULT_PARAMS, **supplied}
        for name, (low, high) in PARAMETER_BOUNDS.items():
            value = values[name]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value)) or not low <= float(value) <= high):
                raise ValueError(f"Precision parameter outside its finite bounds: {name}.")
            if name == "pulse_particles" and int(value) != value:
                raise ValueError("pulse_particles must be an integer.")
            values[name] = int(value) if name == "pulse_particles" else float(value)
        self.params = values

    def save(self, path, metadata=None):
        checkpoint = {
            "schema_version": 1, "policy_kind": POLICY_KIND,
            "parameters": dict(self.params),
            "observation_units": {"volume": "mL", "flow": "mL/s", "tilt": "degrees"},
            "termination": "nearest_requested_volume_particle_count_plus_or_minus_one",
            "metadata": {} if metadata is None else metadata,
        }
        content = json.dumps(checkpoint, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
        return checkpoint

    @classmethod
    def load(cls, path):
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != 1 or checkpoint.get("policy_kind") != POLICY_KIND:
            raise ValueError("Unsupported precision-policy checkpoint.")
        json.dumps(checkpoint, allow_nan=False)
        if set(checkpoint.get("parameters", {})) != set(DEFAULT_PARAMS):
            raise ValueError("Precision checkpoint must contain all precision parameters.")
        return cls(checkpoint["parameters"])


class TargetPourController(_BASE.TargetPourController):
    """Coarse pour, local brake, settled measurement and bounded fine pulses.

    Compatible with PourTrainingEnv(control=precision_control). The environment
    owns the unchanged solver and physical clock. Counts use fixed initial tube
    labels and privileged geometry; they are not camera or real-scale readings.
    """

    MAX_PULSES = 20
    RETURN_BY_SECONDS = 88.0
    HOLD_STABLE_SECONDS = 0.4
    MAX_RETRACT_ATTEMPTS = 5

    def __init__(self, geometry, target_ml=30.0, spacing=0.0035, policy=None):
        if policy is None:
            policy = PrecisionPolicy()
        elif isinstance(policy, dict):
            policy = PrecisionPolicy(policy)
        elif getattr(policy, "name", None) == POLICY_KIND and hasattr(policy, "params"):
            policy = PrecisionPolicy(policy.params)
        else:
            raise ValueError("This controller requires a PrecisionPolicy, not an old stop policy.")
        super().__init__(geometry, target_ml=target_ml, spacing=spacing, policy=policy)

    def reset(self, target_ml=None):
        super().reset(target_ml)
        self.nearest_target_count = int(math.floor(self.target_ml / self.particle_ml + 0.5))
        self._phase_since = 0.0
        self._release_stable_since = 0.0
        self._received_stable_since = 0.0
        self._received_count = 0
        self._released_count = 0
        self._pulse_count = 0
        self._pulse_start_released = 0
        self._pulse_start_committed = 0
        self._pulse_budget = 1
        self._pulse_first_release_source = None
        self._onset_source = None
        self._approach_goal = None
        self._retract_goal = None
        self._retract_attempts = 0
        self._precision_failure = None
        self._last_event = "reset"
        self._last_event_time = 0.0
        self._probe_limit = self.max_source - 0.01
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self.policy.params),
                           nearest_target_count=self.nearest_target_count, pulse_count=0,
                           precision_failure=None, target_band_reached=False)
        return dict(self.status)

    @staticmethod
    def _evaluate(segment, t):
        # Commit exactly the requested acceleration, then the resulting
        # velocity. Reaching a goal cannot teleport position or zero velocity.
        t0, s0, v0, acceleration, horizon, _ = segment
        elapsed = max(0.0, float(t) - t0)
        h = min(elapsed, horizon)
        velocity = v0 + acceleration * h
        source = s0 + v0 * h + 0.5 * acceleration * h * h
        source += velocity * max(0.0, elapsed - horizon)
        return source, velocity

    def _safe_command(self, t, goal, max_rate, dt):
        source, _ = self._evaluate(self._segments[-1], t)
        distance = abs(goal - source)
        # A conservative braking envelope avoids the old late braking followed
        # by a goal snap. Within the arrival band, smoothly brake to rest.
        braking_rate = 0.5 * math.sqrt(2 * self.max_acceleration * distance)
        rate = min(float(max_rate), braking_rate) if distance > 1e-5 else 0.0
        super()._command(t, goal, rate, dt)
        return rate

    def _enter(self, phase, t, event):
        self.phase = phase
        self._phase_since = float(t)
        self._last_event = event
        self._last_event_time = float(t)

    def _retract(self, phase, t, source, event):
        retract = self.policy.params["fine_retract_source"] if phase == "fine_retract" else self.policy.params["retract_source"]
        self._retract_goal = max(self.upright_source, source - retract)
        self._enter(phase, t, event)

    def _return(self, t, reason=None):
        if reason is not None and self._precision_failure is None:
            self._precision_failure = reason
        self._enter("return", t, "return_upright" if reason is None else reason)

    def _start_pulse(self, t, source, released_count, committed_count, deficit):
        self._pulse_count += 1
        self._pulse_start_released = released_count
        self._pulse_start_committed = committed_count
        self._pulse_budget = min(self.policy.params["pulse_particles"], max(1, deficit - 1))
        self._pulse_first_release_source = None
        self._retract_attempts = 0
        # Live pilots showed that fast re-approach can release many particles
        # after the first observed one. Every fine pulse now starts quasistatic.
        self._approach_goal = source
        self._enter("fine_probe", t, "start_fine_pulse")

    def update(self, t, positions, labels, velocities=None):
        """Observe live particles and commit one bounded trajectory command."""
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Physical time must be finite and nonnegative.")
        if self._last_t is not None and t <= self._last_t:
            if abs(t - self._last_t) < 1e-9:
                return dict(self.status)
            raise ValueError("Physical time must increase; reset before restarting.")
        points = np.asarray(positions, dtype=np.float64)
        groups = np.asarray(labels)
        if (points.ndim != 2 or points.shape[1] != 3 or groups.shape != (len(points),)
                or not np.isfinite(points).all()):
            raise ValueError("positions must be finite Nx3 with matching fixed labels.")
        dt = t - self._last_t if self._last_t is not None else 1.0 / 60.0
        if velocities is None:
            velocity = ((points - self._last_points) / dt if self._last_points is not None
                        else np.zeros_like(points))
        else:
            velocity = np.asarray(velocities, dtype=np.float64)
            if velocity.shape != points.shape or not np.isfinite(velocity).all():
                raise ValueError("velocities must be finite and match positions.")
        source = self.source_time(t)
        g = self.geometry
        _, angle = g.mouth_pose(source)
        origin, _ = g.tube_pose(source)
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
        catch = np.sum((impact_xy - center[:2]) ** 2, axis=1) <= (g.BEAKER_RADIUS + 0.5 * self.spacing) ** 2
        catch |= ((points[:, 2] <= rim_z)
                  & (np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2))
        received_count = int(np.count_nonzero(initial_tube & beaker))
        released_count = int(np.count_nonzero(initial_tube & ~tube))
        airborne_count = int(np.count_nonzero(airborne))
        catch_count = int(np.count_nonzero(airborne & catch))
        spilled_count = int(np.count_nonzero(free & ~airborne))
        received = received_count * self.particle_ml
        released = released_count * self.particle_ml
        inflight = catch_count * self.particle_ml
        instant_flow = max(0.0, (released - self._last_released) / dt)
        self._flow += (1 - math.exp(-dt / 0.12)) * (instant_flow - self._flow)
        if received_count != self._received_count:
            self._received_stable_since = t
        if released_count != self._released_count:
            self._release_stable_since = t
        committed_count = received_count + catch_count
        conservative_committed_count = received_count + airborne_count
        deficit = self.nearest_target_count - received_count
        target_band = abs(deficit) <= 1
        quiet = (airborne_count == 0 and t - self._received_stable_since >= self.HOLD_STABLE_SECONDS
                 and t - self._release_stable_since >= self.HOLD_STABLE_SECONDS
                 and self._flow <= 0.10)
        if self.phase == "prepare" and source >= self.pretilt_source - 1e-5:
            self._enter("coarse_pour", t, "pretilt_reached")
        if (self.phase not in ("return", "settle", "done")
                and (t >= self.RETURN_BY_SECONDS or spilled_count > 0)):
            self._return(t, "controller_time_limit" if t >= self.RETURN_BY_SECONDS else "source_spill")
        if self.phase == "coarse_pour":
            reserve_count = int(math.ceil(self.policy.params["reserve_ml"] / self.particle_ml))
            if committed_count >= self.nearest_target_count - reserve_count:
                self._onset_source = source
                self._retract_attempts = 1
                self._retract("coarse_retract", t, source, "coarse_reserve_reached")
        if self.phase in ("fine_approach", "fine_probe"):
            new_released = max(0, released_count - self._pulse_start_released)
            new_committed = max(0, conservative_committed_count - self._pulse_start_committed)
            if new_released > 0 and self._pulse_first_release_source is None:
                self._pulse_first_release_source = source
                self._onset_source = source
            if (new_released >= self._pulse_budget or new_committed >= self._pulse_budget
                    or conservative_committed_count >= self.nearest_target_count - 1):
                self._retract_attempts = 1
                self._retract("fine_retract", t, source, "fine_release_budget_reached")
            elif self.phase == "fine_approach" and source >= self._approach_goal - 1e-5:
                self._enter("fine_probe", t, "slow_probe_started")
            elif source >= self._probe_limit - 1e-5:
                self._return(t, "probe_path_exhausted")
        if self.phase in ("coarse_retract", "fine_retract") and source <= self._retract_goal + 1e-5:
            self._enter("coarse_hold" if self.phase == "coarse_retract" else "fine_hold",
                        t, "local_retraction_reached")
        if self.phase in ("coarse_hold", "fine_hold"):
            if quiet and t - self._phase_since >= self.HOLD_STABLE_SECONDS:
                if target_band:
                    self._return(t)
                elif deficit < -1:
                    self._return(t, "irreversible_overdose")
                elif self._pulse_count >= self.MAX_PULSES:
                    self._return(t, "pulse_limit")
                else:
                    self._start_pulse(t, source, released_count, conservative_committed_count, deficit)
            elif t - self._phase_since >= 1.2:
                if self._retract_attempts >= self.MAX_RETRACT_ATTEMPTS or source <= self.upright_source + 1e-5:
                    self._return(t, "local_flow_did_not_stop")
                else:
                    phase = "coarse_retract" if self.phase == "coarse_hold" else "fine_retract"
                    self._retract_attempts += 1
                    self._retract(phase, t, source, "further_local_retraction")
        if self.phase == "return" and source <= self.upright_source + 1e-5:
            self._enter("settle", t, "upright_reached")
        if self.phase == "settle" and t - self._phase_since >= 1.2 and quiet:
            if (deficit > 1 and self._precision_failure is None
                    and self._pulse_count < self.MAX_PULSES and t < self.RETURN_BY_SECONDS):
                self._start_pulse(t, source, released_count, conservative_committed_count, deficit)
            else:
                if not target_band and self._precision_failure is None:
                    self._precision_failure = "final_dose_outside_particle_band"
                self.done = True
                self._enter("done", t, "settled_upright_complete")

        if self.phase == "prepare":
            goal, rate = self.pretilt_source, 1.0
        elif self.phase == "coarse_pour":
            tilt = math.degrees(float(angle))
            rate = 0.05 if tilt < 80 else 0.025
            if self.target_ml - (received + inflight) < 8:
                rate = min(rate, 0.012)
            goal = self.max_source
        elif self.phase == "fine_approach":
            goal, rate = self._approach_goal, 0.08
        elif self.phase == "fine_probe":
            goal, rate = self._probe_limit, self.policy.params["probe_rate"]
        elif self.phase in ("coarse_retract", "fine_retract"):
            goal, rate = self._retract_goal, 1.0
        elif self.phase in ("coarse_hold", "fine_hold"):
            goal, rate = self._retract_goal, 0.0
        else:
            goal, rate = self.upright_source, 1.0
        rate = self._safe_command(t, goal, rate, dt)
        event = self._last_event
        action = {"stop": self.phase in ("coarse_retract", "fine_retract", "return", "settle", "done"),
                  "rate": float(rate), "predicted_final_ml": (received + inflight),
                  "decision": self.phase, "event": event,
                  "event_this_frame": bool(self._last_event_time == t),
                  "pulse_count": self._pulse_count, "pulse_budget_particles": self._pulse_budget,
                  "new_released_particles": max(0, released_count - self._pulse_start_released),
                  "nearest_target_count": self.nearest_target_count,
                  "remaining_target_particles": deficit, "settled_measurement": bool(quiet),
                  "source_airborne_particles": airborne_count,
                  "precision_failure": self._precision_failure}
        self.status = dict(
            phase=self.phase, done=self.done, target_ml=self.target_ml,
            received_ml=received, released_ml=released, inflight_ml=inflight,
            flow_ml_s=float(self._flow), instantaneous_flow_ml_s=float(instant_flow),
            predicted_final_ml=received + inflight, source_time=source,
            tilt_deg=math.degrees(float(angle)), error_ml=received - self.target_ml,
            particle_count=len(points), tube_remaining_ml=float(np.count_nonzero(initial_tube & tube) * self.particle_ml),
            spilled_ml=spilled_count * self.particle_ml, trim_count=0,
            policy_name=POLICY_KIND, policy_params=dict(self.policy.params), policy_action=action,
            pulse_count=self._pulse_count, pulse_budget_particles=self._pulse_budget,
            source_airborne_particles=airborne_count, nearest_target_count=self.nearest_target_count,
            nearest_attainable_ml=self.nearest_target_count * self.particle_ml,
            target_band_reached=bool(target_band), precision_failure=self._precision_failure,
            precision_success=bool(self.done and target_band and quiet and self._precision_failure is None), last_event=event,
            **self._last_command)
        self._last_t = t
        self._last_points = points.copy()
        self._last_released = released
        self._last_received = received
        self._received_count = received_count
        self._released_count = released_count
        return dict(self.status)
