"""V10 engineering controller: actual settled volume and small angular steps.

This new experiment reuses only the frozen lift-path motion primitives.  It
does not call the V8/V9 state machine, alter particles, or predict a micro dose
from flow times a future duration.  Small steps are complete C2 rest-to-rest
motions followed by quiet measurements.  An exit which subsequently reenters
the tube is still an exit event and forces a measurement before another step.

The coarse cut immediately uses the original joint-safe fast return to78.
That dose tail is logged separately and never used as a slow fine-return
reserve.  Fine corrections measure their own return tail and verify the
actual result after returning.  Tail repeatability and step dose are
experimental, not guaranteed valves.
"""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_control_v2.py"
BASE_SHA256 = "6b17db742e812777d562e6cd15f2c53ee5009fcf75c60052b702d4e5707c95ca"
if hashlib.sha256(BASE_FILE.read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError("V10 motion base differs from the frozen, independently checked source.")
_SPEC = importlib.util.spec_from_file_location("range_v10_frozen_motion_base", BASE_FILE)
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

POLICY_KIND = "settled_event_latched_small_angle_range_control_v10"
PARAMETER_NAMES = base.PARAMETER_NAMES


def static_retained_capacity_ml(tilt_deg, radius_m=.0135, length_m=.18, intervals=4000):
    """Geometric static capacity below the lowest lip, not a PBF prediction.

    The tube cavity has its original hemispherical bottom.  Its cross-section
    radius varies below z=R; using a full cylinder would overestimate capacity.
    Simpson integration is in metres and converted once to millilitres.
    """
    theta = math.radians(float(tilt_deg))
    if not 0. < theta < math.pi / 2 or intervals < 2 or intervals % 2:
        raise ValueError("Static proposal requires 0<tilt<90 degrees and even integration intervals.")
    z = np.linspace(0., float(length_m), int(intervals) + 1)
    radii = np.where(z < radius_m,
        np.sqrt(np.maximum(0., radius_m ** 2 - (z - radius_m) ** 2)), radius_m)
    cutoff = radius_m + (z - length_m) / math.tan(theta)
    area = np.zeros_like(z)
    full = cutoff <= -radii
    area[full] = math.pi * radii[full] ** 2
    partial = (cutoff > -radii) & (cutoff < radii) & (radii > 0)
    r, x = radii[partial], cutoff[partial]
    area[partial] = r ** 2 * np.arccos(np.clip(x / r, -1., 1.)) - x * np.sqrt(np.maximum(0., r ** 2 - x ** 2))
    integral = (length_m / intervals / 3) * (area[0] + area[-1]
        + 4 * np.sum(area[1:-1:2]) + 2 * np.sum(area[2:-1:2]))
    return float(integral * 1e6)


def static_onset_proposal_deg(remaining_ml, radius_m=.0135, length_m=.18):
    """Continuous inverse capacity used only for the next approach angle."""
    remaining_ml = float(remaining_ml)
    if not math.isfinite(remaining_ml) or remaining_ml <= 0:
        raise ValueError("A positive finite measured remaining volume is required.")
    low, high = 1., 89.999
    if remaining_ml > static_retained_capacity_ml(low, radius_m, length_m):
        raise ValueError("Remaining particles exceed the geometric cavity capacity.")
    for _ in range(44):
        middle = (low + high) / 2
        if static_retained_capacity_ml(middle, radius_m, length_m) > remaining_ml:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def capacity_below_observed_plane_ml(surface_world_z, tube_origin, tube_rotation,
                                     radius_m=.0135, length_m=.18, intervals=4000):
    """Geometric occupancy below a horizontal observed particle-centre plane.

    Full-R maps local (x,y,z) to world height.  A circular cross-section allows
    its radial world-Z coefficients to be combined by their Euclidean norm.
    This estimates a pose proposal only; mass/volume truth stays count*q.
    """
    rotation = np.asarray(tube_rotation, dtype=float)
    origin = np.asarray(tube_origin, dtype=float)
    radial = float(np.linalg.norm(rotation[2, :2]))
    if radial < 1e-8 or not math.isfinite(float(surface_world_z)):
        raise ValueError("Observed surface proposal requires a finite tilted tube pose.")
    z = np.linspace(0., float(length_m), int(intervals) + 1)
    radii = np.where(z < radius_m,
        np.sqrt(np.maximum(0., radius_m ** 2 - (z - radius_m) ** 2)), radius_m)
    cutoff = (origin[2] + rotation[2, 2] * z - float(surface_world_z)) / radial
    area = np.zeros_like(z)
    full = cutoff <= -radii
    area[full] = math.pi * radii[full] ** 2
    partial = (cutoff > -radii) & (cutoff < radii) & (radii > 0)
    r, x = radii[partial], cutoff[partial]
    area[partial] = r ** 2 * np.arccos(np.clip(x / r, -1., 1.)) - x * np.sqrt(np.maximum(0., r ** 2 - x ** 2))
    integral = (length_m / intervals / 3) * (area[0] + area[-1]
        + 4 * np.sum(area[1:-1:2]) + 2 * np.sum(area[2:-1:2]))
    return float(integral * 1e6)


class CorrectionPolicy:
    """Fixed engineering profile; the runner may search only coarse speed.

    Angular steps and speeds are degrees and degrees / physical second.
    The legacy constructor options are accepted explicitly for runner API
    compatibility, but unsupported alternative modes are rejected.
    """

    name = POLICY_KIND

    def __init__(self, coarse, deadline_s=24., topup_tilt_deg_s=2.2,
                 topup_lead_s=.12, enable_topup=True, particle_trim=True,
                 fine_step_deg=.003, middle_step_deg=.03, bulk_step_deg=.15,
                 fine_tilt_deg_s=.25, middle_tilt_deg_s=.6,
                 quiet_seconds=.25, flowing_hold_seconds=.35,
                 local_backoff_deg=.15, local_backoff_deg_s=1.,
                 return_tilt_deg_s=4., approach_tilt_deg_s=4.,
                 approach_margin_deg=.03, initial_fine_tilt_deg=80.,
                 onset_margin_deg=.10,
                 surface_quantile=.995,
                 fine_return_calibration_margin_ml=.35, maximum_return_cycles=4,
                 maximum_steps=80):
        self.coarse = dict(base.DirectPolicy(coarse).params)
        values = dict(deadline_s=deadline_s, topup_tilt_deg_s=topup_tilt_deg_s,
                      topup_lead_s=topup_lead_s, fine_step_deg=fine_step_deg,
                      middle_step_deg=middle_step_deg, bulk_step_deg=bulk_step_deg,
                      fine_tilt_deg_s=fine_tilt_deg_s,
                      middle_tilt_deg_s=middle_tilt_deg_s,
                      quiet_seconds=quiet_seconds,
                      flowing_hold_seconds=flowing_hold_seconds,
                      local_backoff_deg=local_backoff_deg,
                      local_backoff_deg_s=local_backoff_deg_s,
                      return_tilt_deg_s=return_tilt_deg_s,
                      approach_tilt_deg_s=approach_tilt_deg_s,
                      approach_margin_deg=approach_margin_deg,
                      initial_fine_tilt_deg=initial_fine_tilt_deg,
                      onset_margin_deg=onset_margin_deg,
                      surface_quantile=surface_quantile,
                      fine_return_calibration_margin_ml=fine_return_calibration_margin_ml)
        bounds = dict(deadline_s=(24., 24.), topup_tilt_deg_s=(.1, 4.),
                      topup_lead_s=(0., 1.), fine_step_deg=(.001, .02),
                      middle_step_deg=(.01, .2), bulk_step_deg=(.05, .5),
                      fine_tilt_deg_s=(.05, .5), middle_tilt_deg_s=(.1, 2.),
                      quiet_seconds=(.2, .6), flowing_hold_seconds=(.25, 1.),
                      local_backoff_deg=(.025, .5), local_backoff_deg_s=(.1, 2.),
                      return_tilt_deg_s=(1., 6.), approach_tilt_deg_s=(1., 6.),
                      approach_margin_deg=(.005, 1.), initial_fine_tilt_deg=(78., 82.),
                      onset_margin_deg=(.01, .5),
                      surface_quantile=(.95, 1.),
                      fine_return_calibration_margin_ml=(.1, 1.))
        for key, value in values.items():
            if (isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value))
                    or not bounds[key][0] <= float(value) <= bounds[key][1]):
                raise ValueError("V10 profile parameter outside finite bounds: " + key)
            setattr(self, key, float(value))
        for key, value, low, high in (("maximum_return_cycles", maximum_return_cycles, 1, 8),
                                      ("maximum_steps", maximum_steps, 1, 200)):
            if isinstance(value, bool) or int(value) != value or not low <= value <= high:
                raise ValueError("V10 integer profile parameter outside bounds: " + key)
            setattr(self, key, int(value))
        self.enable_topup = bool(enable_topup)
        if not particle_trim:
            raise ValueError("The V10 engineering controller requires settled small-step correction.")
        self.particle_trim = True
        self.params = {**self.coarse, **{key: getattr(self, key) for key in values},
                       "enable_topup": self.enable_topup, "particle_trim": True,
                       "maximum_return_cycles": self.maximum_return_cycles,
                       "maximum_steps": self.maximum_steps,
                       "relative_dose_tolerance": .001,
                       "fine_tail_prediction_used": False,
                       "fine_approach_basis": "observed_world_Z_particle_centre_quantile_calibrated_static_capacity",
                       "initial_fine_tilt_deg_and_approach_margin_are_legacy_compatibility_only": True,
                       "topup_lead_s_is_unused_compatibility_argument": True}


class TargetPourController(base.TargetPourController):
    """Read-only particle observations plus independently committed motions."""

    def __init__(self, geometry, target_ml=30., spacing=.0027, policy=None):
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("V10 requires CorrectionPolicy; old policies are not interchangeable.")
        self.profile = policy
        self.deadline = policy.deadline_s
        super().__init__(geometry, target_ml, spacing, base.DirectPolicy(policy.coarse))

    def reset(self, target_ml=None):
        result = super().reset(target_ml)
        if not 20. <= self.target_ml <= 40.:
            raise ValueError("This range experiment supports requested targets in [20,40] mL.")
        self._previous_inside = None
        self._fixed_labels = None
        self._ever_exited = None
        self._exit_event_count = 0
        self._step_exit_latched = False
        self._step_new_exit_events = 0
        self._quiet_since = None
        self._stationary_since = None
        self._hold_destination = None
        self._brake_destination = None
        self._coarse_cut_angle = 78.
        self._last_flow_angle = None
        self._onset_proposal = None
        self._return_start_received = None
        self._return_start_released = None
        self._active_return_profile = None
        self._fine_tail_samples = []
        self._coarse_tail_samples = []
        self._tail_samples = []
        self._tail_reserve_particles = 0
        self._return_cycles = 0
        self._steps = 0
        self._correction_count = 0
        self._pulse_start_received = None
        self._pulse_start_released = None
        self._pulse_results = []
        self._pending_pulse_kind = None
        self._pending_end = None
        self._last_step_evidence = {}
        self._last_event = "prepare_clock_started"
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
                           correction_enabled=self.profile.enable_topup, deadline_s=self.deadline)
        return result

    def _evaluate(self, segment, t):
        t0, u0, v0, acceleration, horizon, goal, kind = segment
        if kind == "v10_C2_source_step":
            elapsed = float(t) - t0
            if elapsed >= horizon:
                return float(goal), 0., 0., "v10_C2_step_endpoint_hold"
            if elapsed <= 0:
                return float(u0), 0., 0., kind
            x = elapsed / horizon
            delta = goal - u0
            progress = x ** 3 * (10 - 15 * x + 6 * x * x)
            first = 30 * x * x * (1 - x) ** 2
            second = 60 * x * (1 - x) * (1 - 2 * x)
            return (float(u0 + delta * progress), float(delta * first / horizon),
                    float(delta * second / horizon ** 2), kind)
        if kind == "v10_stationary_hold":
            return float(u0), 0., 0., kind
        return super()._evaluate(segment, t)

    def _commit(self, segment):
        t = float(segment[0])
        if self._times[-1] == t:
            self._segments[-1] = segment
        else:
            self._times.append(t)
            self._segments.append(segment)

    def _hold(self, t, u, destination):
        self._commit((float(t), float(u), 0., 0., 0., float(u), "v10_stationary_hold"))
        self.phase = destination
        self._quiet_since = None
        self._stationary_since = float(t)
        self._last_command = {"command_goal_source": float(u), "command_max_rate": 0.,
                              "command_source_velocity": 0., "command_next_source_velocity": 0.,
                              "command_acceleration": 0., "command_horizon_s": 0.,
                              "motion_continuity": "stationary after zero physical source velocity"}

    def _duration_for_step(self, u0, u1, angle_rate):
        angle_delta = abs(float(self.geometry.tilt_deg(u1) - self.geometry.tilt_deg(u0)))
        duration = max(.1, 1.875 * angle_delta / angle_rate, 1.875 * abs(u1 - u0))
        limits = self.geometry.JOINT_VELOCITY_LIMITS_RAD_S
        for _ in range(30):
            segment = (0., u0, 0., 0., duration, u1, "v10_C2_source_step")
            velocity_ratio = acceleration_peak = u_rate_peak = tilt_rate_peak = 0.
            # At least 240 Hz, plus 81 phase samples for short steps.
            for moment in np.linspace(0., duration, max(81, int(math.ceil(duration * 240)) + 1)):
                u, rate, acceleration, _ = self._evaluate(segment, moment)
                first, second = self.geometry.joint_derivative(u), self.geometry.joint_second_derivative(u)
                velocity_ratio = max(velocity_ratio, float(np.max(np.abs(first * rate) / limits)))
                acceleration_peak = max(acceleration_peak,
                    float(np.max(np.abs(second * rate ** 2 + first * acceleration))))
                u_rate_peak = max(u_rate_peak, abs(rate))
                tilt_rate_peak = max(tilt_rate_peak,
                    abs(math.degrees(float(self.geometry.angle_derivative(u))) * rate))
            if (velocity_ratio <= .97 and acceleration_peak <= 23. and u_rate_peak <= 1.
                    and tilt_rate_peak <= angle_rate * (1 + 1e-10)):
                return duration, {"planned_joint_velocity_limit_ratio": velocity_ratio,
                                  "planned_joint_acceleration_rad_s2": acceleration_peak,
                                  "planned_u_rate_peak": u_rate_peak,
                                  "planned_tilt_speed_peak_deg_s": tilt_rate_peak}
            duration *= max(1.05, velocity_ratio / .97,
                            math.sqrt(acceleration_peak / 23.), u_rate_peak,
                            tilt_rate_peak / angle_rate)
        raise RuntimeError("No bounded-duration C2 step passes the actual joint motion limits.")

    def _start_step(self, t, u, target_angle, rate, phase):
        if abs(self.motion_state(t)[1]) > 1e-8:
            raise RuntimeError("A C2 rest-to-rest step cannot start from moving source coordinates.")
        target_angle = float(np.clip(target_angle, 78., 125.))
        goal = float(self.geometry.u_for_tilt(target_angle))
        duration, evidence = self._duration_for_step(u, goal, rate)
        self._commit((float(t), float(u), 0., 0., duration, goal, "v10_C2_source_step"))
        self.phase, self._pending_end = phase, float(t) + duration
        self._step_exit_latched, self._step_new_exit_events = False, 0
        self._last_step_evidence = {"from_tilt_deg": float(self.geometry.tilt_deg(u)),
            "to_tilt_deg": target_angle, "physical_duration_s": duration,
            "angular_speed_cap_deg_s": float(rate), **evidence}
        self._last_command = {"command_goal_source": goal, "command_max_rate": evidence["planned_u_rate_peak"],
            "command_source_velocity": 0., "command_next_source_velocity": 0.,
            "command_acceleration": 0., "command_horizon_s": duration,
            "motion_continuity": "complete C2 rest-to-rest source step; pure endpoint hold",
            **evidence}

    def _home_budget(self):
        return self.geometry.PREPARE_END_U / self.policy.params["return_fraction"] + .45

    def _remaining_return_time(self, u):
        if u <= self.geometry.PREPARE_END_U + 1e-10:
            return self._home_budget()
        duration, _ = self._duration_for_step(u, self.geometry.PREPARE_END_U,
                                               self.profile.return_tilt_deg_s)
        return duration + self.profile.quiet_seconds + self._home_budget() + .10

    def _begin_return(self, t, u, received_count, released_count, reason):
        self._return_start_received = int(received_count)
        self._return_start_released = int(released_count)
        self._return_reason = reason
        self._return_cycles += 1
        self._active_return_profile = "fine_C2_slow_return"
        self._start_step(t, u, 78., self.profile.return_tilt_deg_s, "measured_return_78")
        self._last_event = "measured_slow_return_to_78_" + reason

    def _fast_coarse_return_command(self, t, u, velocity, dt):
        """The original joint-safe immediate cut, without stopping at 90deg."""
        desired = -self.policy.params["return_fraction"]
        acceleration, evidence = self._select_acceleration(u, velocity, desired, dt)
        segment = (float(t), u, velocity, acceleration, dt,
                   self.geometry.PREPARE_END_U, "C1_retract_to_78_commit")
        self._commit(segment)
        self._last_command = {"command_goal_source": segment[5], "command_max_rate": abs(desired),
            "command_source_velocity": velocity, "command_next_source_velocity": velocity + acceleration * dt,
            "command_acceleration": acceleration, "command_horizon_s": dt,
            "planned_joint_velocity_limit_ratio": evidence[0],
            "planned_joint_acceleration_rad_s2": evidence[1],
            "motion_continuity": "original joint-safe immediate coarse cut; C2 only at78 anchor"}

    def _record_pulse(self, t, received_count, released_count):
        if self._pulse_start_received is None:
            return
        self._pulse_results.append({"kind": self._pending_pulse_kind,
            "settled_physical_time_s": float(t),
            "received_delta_particles": int(received_count - self._pulse_start_received),
            "net_out_delta_particles": int(released_count - self._pulse_start_released),
            "inside_to_outside_events_during_step": self._step_new_exit_events,
            "exit_event_latched": self._step_exit_latched})
        self._pulse_start_received = None

    def _start_home(self, t, u):
        if abs(u - self.geometry.PREPARE_END_U) > 1e-9:
            raise RuntimeError("Home starts only at the exact physical 78-degree C2 anchor.")
        self.phase = "home_return"
        self._start_reverse_prepare_clock(t, u)
        self._last_event = "exact_reverse_prepare_clock_started"

    def update(self, t, positions, labels, velocities=None):
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Finite nonnegative raw physical time is required.")
        if self._last_t is not None and t <= self._last_t:
            if abs(t - self._last_t) < 1e-9:
                return dict(self.status)
            raise ValueError("Reset before reversing raw physical time.")
        points, groups = np.asarray(positions, dtype=float), np.asarray(labels)
        if (points.ndim != 2 or points.shape[1] != 3 or groups.shape != (len(points),)
                or not np.isfinite(points).all() or np.any((groups != 0) & (groups != 1))):
            raise ValueError("Finite Nx3 positions with fixed 0/1 particle labels are required.")
        if self._fixed_labels is None:
            self._fixed_labels = groups.copy()
            self._ever_exited = np.zeros(len(points), dtype=bool)
        elif not np.array_equal(groups, self._fixed_labels):
            raise ValueError("Particle identities and labels must remain fixed.")
        dt = t - self._last_t if self._last_t is not None else 1 / 60
        velocity_points = ((points - self._last_points) / dt if self._last_points is not None else
                           np.zeros_like(points)) if velocities is None else np.asarray(velocities, dtype=float)
        if velocity_points.shape != points.shape or not np.isfinite(velocity_points).all():
            raise ValueError("Finite matching particle velocities are required.")
        g = self.geometry
        u, u_velocity, u_acceleration, continuation = self.motion_state(t)
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
                  & (delta[:, 2] >= 0.) & (delta[:, 2] <= g.BEAKER_HEIGHT))
        source = groups == 1
        inside = source & tube
        free = source & ~(tube | beaker)
        airborne = free & (points[:, 2] > g.TABLE_TOP_Z + self.spacing)
        received_count = int(np.count_nonzero(source & beaker))
        released_count = int(np.count_nonzero(source & ~tube))
        received, released = received_count * self.particle_ml, released_count * self.particle_ml
        airborne_count = int(np.count_nonzero(airborne))
        new_exit = (np.zeros(len(points), dtype=bool) if self._previous_inside is None else
                    self._previous_inside & source & ~tube)
        new_exit_count = int(np.count_nonzero(new_exit))
        new_first_exit = new_exit & ~self._ever_exited
        self._ever_exited |= new_exit
        self._exit_event_count += new_exit_count
        if new_exit_count and self.phase not in ("prepare", "home_return", "settle", "done"):
            # Recoil/78-degree residual exits are not evidence of a new
            # forward-flow onset.  Keep the highest observed forward angle so
            # a late low-angle exit cannot erase the reapproach reference.
            if self.phase in ("fine_step", "fine_approach"):
                angle = float(g.tilt_deg(u))
                self._last_flow_angle = angle if self._last_flow_angle is None else max(self._last_flow_angle, angle)
            self._step_exit_latched = True
            self._step_new_exit_events += new_exit_count
        instantaneous_flow = max(0., (released - self._last_released) / dt)
        self._flow += (1 - math.exp(-dt / .06)) * (instantaneous_flow - self._flow)
        rim_z = center[2] + g.BEAKER_HEIGHT
        fall = (velocity_points[:, 2] + np.sqrt(velocity_points[:, 2] ** 2
                + 2 * 9.81 * np.maximum(points[:, 2] - rim_z, 0))) / 9.81
        impact = points[:, :2] + velocity_points[:, :2] * fall[:, None]
        catch = np.sum((impact - center[:2]) ** 2, axis=1) <= (g.BEAKER_RADIUS + .5 * self.spacing) ** 2
        catch |= (points[:, 2] <= rim_z) & (np.sum(delta[:, :2] ** 2, axis=1) <= g.BEAKER_RADIUS ** 2)
        inflight = int(np.count_nonzero(airborne & catch)) * self.particle_ml
        coarse_action = self.policy.act({"target_ml": self.target_ml, "received_ml": received,
                                         "inflight_ml": inflight, "flow_ml_s": self._flow})
        band = self.target_ml * .001
        shortfall = self.target_ml - received
        changes = (received_count != self._last_received_count or released_count != self._last_released_count
                   or new_exit_count > 0)
        stationary = abs(u_velocity) < 1e-9
        if stationary and airborne_count == 0 and not changes:
            if self._quiet_since is None:
                self._quiet_since = t
        else:
            self._quiet_since = None
        settled = self._quiet_since is not None and t - self._quiet_since >= self.profile.quiet_seconds - 1e-9

        if self.phase == "prepare" and t >= self._prepare_end - 1e-9:
            self.phase = "coarse_dose"
            self._last_event = "prepare_78_anchor_reached"
        if self.phase == "coarse_dose":
            if coarse_action["stop"] or t >= self.deadline - self._home_budget() - 5.:
                self._coarse_cut_angle = float(g.tilt_deg(u))
                self.phase = "coarse_fast_retract"
                self._active_return_profile = "coarse_original_fast_return"
                self._return_start_received = received_count
                self._return_start_released = released_count
                self._return_cycles += 1
                self._return_reason = "coarse_immediate_joint_safe_cut"
                self._last_event = "coarse_immediate_fast_return_without_high_angle_hold"
        if (self.phase == "coarse_fast_retract" and u <= g.PREPARE_END_U + 1e-12
                and stationary):
            self._hold(t, u, "anchor_hold")
            settled = False
            self._last_event = "coarse_fast_return_78_anchor_reached"
        if self.phase == "brake" and stationary:
            self._hold(t, u, self._brake_destination)
            settled = False
            self._last_event = "physical_motion_stopped_before_measurement"

        step_phases = ("fine_step", "fine_approach", "local_backoff", "measured_return_78")
        if self.phase in step_phases and t >= self._pending_end - 1e-9:
            destination = "anchor_hold" if self.phase == "measured_return_78" else "fine_hold"
            self._hold(t, u, destination)
            settled = False
            self._last_event = "complete_C2_step_enter_quiet_measurement"

        if self.phase in ("coarse_hold", "fine_hold"):
            if settled:
                self._record_pulse(t, received_count, released_count)
                reserve_ml = self._tail_reserve_particles * self.particle_ml
                # The only correction gate is the requested strict dose band.
                # No .15/.2-mL eligibility gap and no flow-times-time estimate.
                initial_fine_calibration = (not self._fine_tail_samples
                    and shortfall <= self.profile.fine_return_calibration_margin_ml)
                return_due = (initial_fine_calibration or not self.profile.enable_topup
                    or shortfall <= band + reserve_ml or self._steps >= self.profile.maximum_steps
                    or t + self._remaining_return_time(u) >= self.deadline - .05)
                if return_due:
                    self._begin_return(t, u, received_count, released_count,
                        "initial_fine_tail_calibration" if initial_fine_calibration else "settled_tail_reserved")
                    settled = False
                else:
                    if shortfall - reserve_ml > 1.:
                        step, rate = self.profile.bulk_step_deg, self.profile.topup_tilt_deg_s
                    elif shortfall - reserve_ml > .3:
                        step, rate = self.profile.middle_step_deg, self.profile.middle_tilt_deg_s
                    else:
                        step, rate = self.profile.fine_step_deg, self.profile.fine_tilt_deg_s
                    self._steps += 1
                    self._pulse_start_received, self._pulse_start_released = received_count, released_count
                    self._pending_pulse_kind = "forward_small_angle_step"
                    self._start_step(t, u, min(125., float(g.tilt_deg(u)) + step), rate, "fine_step")
                    self._last_event = "settled_shortfall_small_angle_step"
            elif (stationary and self._stationary_since is not None
                  and t - self._stationary_since >= self.profile.flowing_hold_seconds
                  and g.tilt_deg(u) > 78. + self.profile.local_backoff_deg):
                # A fixed mouth is not a valve: if quiet never occurs, retreat
                # locally and remeasure instead of a multi-degree sweep.
                self._start_step(t, u, float(g.tilt_deg(u)) - self.profile.local_backoff_deg,
                                 self.profile.local_backoff_deg_s, "local_backoff")
                self._last_event = "unsettled_hold_small_reverse_step"

        if self.phase == "anchor_hold" and settled:
            if self._return_start_received is not None:
                tail_particles = received_count - self._return_start_received
                sample = {"cycle": self._return_cycles,
                    "motion_profile": self._active_return_profile,
                    "settled_physical_time_s": t, "received_delta_particles": tail_particles,
                    "net_out_delta_particles": released_count - self._return_start_released,
                    "received_before_return_ml": self._return_start_received * self.particle_ml,
                    "received_after_return_ml": received}
                self._tail_samples.append(sample)
                if self._active_return_profile == "fine_C2_slow_return":
                    self._fine_tail_samples.append(sample)
                    self._tail_reserve_particles = max(0, int(tail_particles))
                else:
                    self._coarse_tail_samples.append(sample)
                self._return_start_received = None
            can_correct = (shortfall > band and self.profile.enable_topup
                and self._return_cycles < self.profile.maximum_return_cycles
                and self._steps < self.profile.maximum_steps
                and t + self._home_budget() + 1. < self.deadline)
            if can_correct:
                self._correction_count += 1
                # Reuse only the past observed geometric onset, then resume
                # bounded steps.  This is motion history, not fluid replay.
                # Static capacity only proposes a geometric approach.  It
                # never supplies a dose, pass flag, or flow-tail prediction.
                remaining_count = int(np.count_nonzero(inside))
                remaining_ml = remaining_count * self.particle_ml
                source_heights = points[inside, 2]
                if len(source_heights) < 2:
                    raise RuntimeError("Insufficient remaining source particles for an observed-plane proposal.")
                observed_surface = float(np.quantile(source_heights, self.profile.surface_quantile))
                effective_ml = capacity_below_observed_plane_ml(observed_surface, origin, rotation,
                                                                 float(g.TUBE_RADIUS), float(g.TUBE_LENGTH))
                onset = static_onset_proposal_deg(effective_ml, float(g.TUBE_RADIUS), float(g.TUBE_LENGTH))
                approach_angle = float(np.clip(onset - self.profile.onset_margin_deg, 78., 89.8))
                self._onset_proposal = {"remaining_inside_particles": remaining_count,
                    "remaining_inside_ml": remaining_ml,
                    "observed_surface_world_z_m": observed_surface,
                    "observed_surface_quantile": self.profile.surface_quantile,
                    "maximum_inside_source_world_z_m": float(np.max(source_heights)),
                    "surface_definition": "linear_quantile_of_inside_source_particle_centres_world_Z_no_radius_offset",
                    "effective_geometric_occupancy_ml": effective_ml,
                    "effective_to_nominal_volume_ratio": effective_ml / remaining_ml,
                    "actual_tube_origin_world_z_m": float(origin[2]),
                    "actual_tube_world_z_axis_coefficients": rotation[2, :].tolist(),
                    "static_onset_tilt_deg": onset, "conservative_margin_deg": self.profile.onset_margin_deg,
                    "approach_tilt_deg": approach_angle,
                    "shape": "original_cylinder_with_hemispherical_bottom",
                    "basis": "geometric_static_capacity_only_not_liquid_or_acceptance_prediction"}
                self._pulse_start_received, self._pulse_start_released = received_count, released_count
                self._pending_pulse_kind = "reapproach_below_previous_observed_flow"
                self._start_step(t, u, approach_angle, self.profile.approach_tilt_deg_s, "fine_approach")
                self._last_event = "strict_undershoot_reapproach_without_deadzone"
            else:
                self._start_home(t, u)

        joints = g.joint_positions(u)
        joint_velocity = g.joint_derivative(u) * u_velocity
        home_position_error = float(np.linalg.norm(origin - self._home_origin))
        home_orientation_error = 2 * math.acos(float(np.clip(abs(np.dot(quaternion, self._home_quaternion)), 0, 1)))
        home_joint_error = float(np.max(np.abs(joints - self._home_q)))
        home_static = bool(u <= 1e-9 and home_position_error <= 1e-6 and home_joint_error <= 1e-8
                           and np.max(np.abs(joint_velocity)) <= 1e-6)
        if self.phase == "home_return" and home_static:
            self.phase = "settle"
            self._last_event = "home_anchor_reached"
        if self.phase in ("settle", "done") and home_static and airborne_count == 0 and not changes:
            if self._stable_since is None:
                self._stable_since = t
            self._stable_frames += 1
        else:
            self._stable_since, self._stable_frames = None, 0
        if (self.phase == "settle" and not self.timed_out and self._stable_frames >= base.STABLE_FRAMES
                and t - self._stable_since >= base.HOME_QUIET_SECONDS - 1e-9 and t <= self.deadline + 1e-9):
            self.phase, self.done, self._completed_at = "done", True, t
            self._last_event = "home_and_actual_measurement_quiet_complete"
        if not self.done and t >= self.deadline - 1e-9:
            self.timed_out = True
            self._last_event = "physical_24s_deadline_timeout_latched"
        if self.timed_out:
            self.done = False

        if self.phase == "coarse_dose":
            derivative = abs(math.degrees(g.angle_derivative(u)))
            desired = min(1., self.policy.params["dose_tilt_deg_s"] / max(derivative, 1e-9))
            self._command(t, u, u_velocity, desired, dt)
        elif self.phase == "coarse_fast_retract":
            self._fast_coarse_return_command(t, u, u_velocity, dt)
        elif self.phase == "brake":
            self._command(t, u, u_velocity, 0., dt)
        elif self.phase in ("done", "settle"):
            # Existing reverse-prepare continuation already holds exact home.
            pass
        # Every C2 step and prepare/home clock is committed in full.  No
        # per-frame replacement, and every future query is observation-free.
        action = {"decision": self.phase, "stop": self.phase not in ("coarse_dose", "fine_step"),
                  "coarse_action": coarse_action, "fine_tail_prediction_used": False,
                  "fine_control_basis": "settled_actual_received_plus_measured_same_episode_return_tail",
                  "tail_reserve_particles": self._tail_reserve_particles,
                  "new_inside_to_outside_events": new_exit_count,
                  "step_exit_event_latched": self._step_exit_latched,
                  "requested_dose_band_ml": band,
                  "dose_in_band_current_measurement": abs(received - self.target_ml) <= band,
                  "quiet_required_s": self.profile.quiet_seconds}
        self.status = {"phase": self.phase, "direct_phase": self.phase, "done": self.done,
            "target_ml": self.target_ml, "received_ml": received, "released_ml": released,
            "inflight_ml": inflight, "flow_ml_s": self._flow,
            "source_airborne_particles": airborne_count, "policy_name": POLICY_KIND,
            "policy_params": dict(self.profile.params), "policy_action": action,
            "timed_out": self.timed_out, "completed_by_deadline": self.done and not self.timed_out,
            "completed_physical_time_s": self._completed_at, "home_static": home_static,
            "home_position_error_m": home_position_error, "home_orientation_error_rad": home_orientation_error,
            "home_joint_error_rad": home_joint_error, "stable_frames": self._stable_frames,
            "required_stable_frames": base.STABLE_FRAMES,
            "home_quiet_seconds": 0. if self._stable_since is None else t - self._stable_since,
            "error_ml": received - self.target_ml, "correction_count": self._correction_count,
            "correction_enabled": self.profile.enable_topup, "deadline_s": self.deadline,
            "mode_event": self._last_event, "scale_ml": received,
            "sensor": "actual_full_R_particle_membership_after_quiet",
            "inflight_used_for_control": self.phase == "coarse_dose",
            "correction_uses_scale_only": True, "source_time": u, "u": u,
            "actual_u_velocity": u_velocity, "actual_u_acceleration": u_acceleration,
            "tilt_deg": float(g.tilt_deg(u)), "joint_positions_rad": joints.tolist(),
            "joint_velocity_rad_s": joint_velocity.tolist(),
            "full_R_containment": True,
            "tube_pose_measurement_basis": "observed_collider_float_pose" if observed is not None else "planned_double_pose",
            "future_continuation": continuation, "inside_to_outside_events": self._exit_event_count,
            "new_inside_to_outside_events": new_exit_count,
            "new_first_exit_particle_count": int(np.count_nonzero(new_first_exit)),
            "ever_exited_particle_count": int(np.count_nonzero(self._ever_exited)),
            "step_exit_event_latched": self._step_exit_latched,
            "quiet_measured_seconds": 0. if self._quiet_since is None else t - self._quiet_since,
            "settled_measurement": settled,
            "small_angle_steps": self._steps, "last_step": dict(self._last_step_evidence),
            "completed_step_doses": list(self._pulse_results),
            "return_tail_samples": list(self._tail_samples),
            "coarse_fast_return_tail_samples": list(self._coarse_tail_samples),
            "fine_slow_return_tail_samples": list(self._fine_tail_samples),
            "return_tail_reserve_particles": self._tail_reserve_particles,
            "forward_flow_reference_tilt_deg": self._last_flow_angle,
            "onset_approach_proposal": self._onset_proposal,
            "return_cycles": self._return_cycles,
            "home_return_uses_exact_reverse_prepare": self._home_clock_started is not None,
            "home_return_clock_start_physical_s": self._home_clock_started,
            **self._last_command}
        self._last_t, self._last_points = t, points.copy()
        self._last_received_count, self._last_released_count = received_count, released_count
        self._last_released = released
        self._previous_inside = inside.copy()
        return dict(self.status)
