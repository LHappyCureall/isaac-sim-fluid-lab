"""Closed-loop top-up on the frozen lift path, with a 10 s or 12 s deadline.

The coarse stop is the existing direct policy. After the 78 degree hold is
quiet, a short extra tilt adds the measured shortfall, then the arm goes home.
This file does not edit the archived five-second controller.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_BASE_PATH = Path(__file__).resolve().parents[2] / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_control_v2.py"
_SPEC = importlib.util.spec_from_file_location("timed_correction_lift_control", _BASE_PATH)
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

POLICY_KIND = "timed_correction_closed_loop_v8"


class CorrectionPolicy:
    """Coarse direct-policy parameters plus one top-up setting."""

    name = POLICY_KIND

    def __init__(self, coarse, deadline_s, topup_tilt_deg_s, topup_lead_s, enable_topup=True,
                 reserve_ml=0.0, lip_pulses=False, scale_feedback=False, dose_duration_s=0.0,
                 scale_pulses=False, chunk_margin_ml=1.0, chunk_tilt_deg_s=3.0,
                 fine_percent=False, particle_trim=False):
        self.coarse = {name: float(coarse[name]) for name in base.PARAMETER_NAMES}
        self.deadline_s = float(deadline_s)
        self.topup_tilt_deg_s = float(topup_tilt_deg_s)
        self.topup_lead_s = float(topup_lead_s)
        self.enable_topup = bool(enable_topup)
        self.reserve_ml = float(reserve_ml)
        # Stay near the pour lip and add a few particles at a time. The 78-degree
        # top-up spends about three seconds climbing back to the stream.
        self.lip_pulses = bool(lip_pulses)
        # Beaker scale: settled mass already in the cup. The coarse pour is an
        # open-loop duration. Inflight particles and flow are not a stop signal.
        self.scale_feedback = bool(scale_feedback)
        self.dose_duration_s = float(dose_duration_s)
        # Coarse stop stays on the short landing. After the cup is quiet, the
        # scale sizes one large addition, then particle-sized trims. Flow is
        # not used to decide those additions.
        self.scale_pulses = bool(scale_pulses)
        self.chunk_margin_ml = float(chunk_margin_ml)
        self.chunk_tilt_deg_s = float(chunk_tilt_deg_s)
        # Aim about 0.15 mL short, then add one particle at a time. One spilled
        # particle is not a failure. The pass band is max(0.1% of target, one particle).
        self.fine_percent = bool(fine_percent)
        # After the fill is quiet at the lip, a hole of a few 0.020 mL particles
        # is closed one particle at a time at 2 deg/s. The fill tail is unchanged.
        self.particle_trim = bool(particle_trim)
        self.params = {
            **self.coarse,
            "deadline_s": self.deadline_s,
            "topup_tilt_deg_s": self.topup_tilt_deg_s,
            "topup_lead_s": self.topup_lead_s,
            "enable_topup": self.enable_topup,
            "reserve_ml": self.reserve_ml,
            "lip_pulses": self.lip_pulses,
            "scale_feedback": self.scale_feedback,
            "dose_duration_s": self.dose_duration_s,
            "scale_pulses": self.scale_pulses,
            "chunk_margin_ml": self.chunk_margin_ml,
            "chunk_tilt_deg_s": self.chunk_tilt_deg_s,
            "fine_percent": self.fine_percent,
            "particle_trim": self.particle_trim,
        }

    def act(self, observation):
        raise RuntimeError("The coarse DirectPolicy owns act().")


class TargetPourController(base.TargetPourController):
    def __init__(self, geometry, target_ml=30.0, spacing=0.0035, policy=None):
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Timed correction requires CorrectionPolicy.")
        self.deadline = policy.deadline_s
        self.topup_tilt_deg_s = policy.topup_tilt_deg_s
        self.topup_lead_s = policy.topup_lead_s
        self.enable_topup = policy.enable_topup
        self.reserve_ml = policy.reserve_ml
        self.lip_pulses = policy.lip_pulses
        self.scale_feedback = policy.scale_feedback
        self.dose_duration_s = policy.dose_duration_s
        self.scale_pulses = policy.scale_pulses
        self.chunk_margin_ml = policy.chunk_margin_ml
        self.chunk_tilt_deg_s = policy.chunk_tilt_deg_s
        self.fine_percent = policy.fine_percent
        self.particle_trim = policy.particle_trim
        self._public_params = dict(policy.params)
        super().__init__(geometry, target_ml=target_ml, spacing=spacing, policy=base.DirectPolicy(policy.coarse))

    def reset(self, target_ml=None):
        super().reset(target_ml)
        self._topup_count = 0
        self._topup_start_released = 0
        self._topup_budget = 0
        self._active_topup_tilt = self.topup_tilt_deg_s
        self._nudge_span = 3.0
        self._bump_budget = 0
        self._lip_floor_tilt = 0.0
        self._nudge_tilt_cap = 0.0
        self._nudge_released = 0
        self._lip_quiet_since = None
        self._lip_phase_started = None
        self._return_from_fill = False
        self._dose_started = None
        self._slow_backoff = False
        self._trim_rounds = 0
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self._public_params),
                           correction_count=0, correction_enabled=self.enable_topup, deadline_s=self.deadline)
        return dict(self.status)

    def _estimate_return_budget(self, u, velocity):
        budget = super()._estimate_return_budget(u, velocity)
        budget["return_deadline_physical_s"] = self.deadline - budget["total_return_reserved_s"]
        return budget

    def _home_budget(self):
        return self.geometry.PREPARE_END_U / self.policy.params["return_fraction"] + 0.45

    def _time_to_home(self, u, velocity):
        try:
            retract = self._retract_duration_estimate(u, velocity)
        except RuntimeError:
            retract = 1.0
        return retract + self._home_budget()

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
            rotation = np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                                 (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                                 (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))
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
        inflight = np.count_nonzero(airborne & catch) * self.particle_ml
        airborne_count = int(np.count_nonzero(airborne))
        instantaneous_flow = max(0.0, (released - self._last_released) / dt)
        self._flow += (1 - math.exp(-dt / .06)) * (instantaneous_flow - self._flow)
        action = self.policy.act({"target_ml": self.target_ml, "received_ml": received,
                                  "inflight_ml": inflight, "flow_ml_s": self._flow})
        if self.phase == "prepare" and t >= self._prepare_end - 1e-9:
            self.phase = "dose"
            self._dose_started = t
            self._mode_event = "prepare_78_anchor_reached"
        if self.phase == "return":
            self.phase = "retract_78" if u >= g.PREPARE_END_U else "home_return"
        if self.phase == "dose":
            self._return_budget = self._estimate_return_budget(u, source_velocity)
        return_no_later = self._return_budget.get("return_deadline_physical_s", self.deadline)
        reserved = self.reserve_ml > 0 and received + inflight >= self.target_ml - self.reserve_ml
        duration_stop = (self.scale_feedback and self._dose_started is not None
                         and t - self._dose_started >= self.dose_duration_s - 1e-9)
        oracle_stop = (not self.scale_feedback) and (action["stop"] or reserved)
        if self.phase == "dose" and (duration_stop or oracle_stop or t >= return_no_later):
            self._return_reason = ("open_loop_dose_duration" if duration_stop else
                                   "reserve_reached" if reserved else
                                   "predicted_target_reached" if action["stop"] else "deadline_forced_return")
            if self.lip_pulses and t < return_no_later:
                self.phase = "lip_backoff"
                self._lip_cut_tilt = float(g.tilt_deg(u))
                self._lip_floor_tilt = self._lip_cut_tilt - 4.0
                self._lip_quiet_since = None
                self._mode_event = "lip_backoff_after_coarse"
            else:
                self.phase = "retract_78"
                self._mode_event = "cut_flow_retract_to_78"
        if self.phase == "lip_backoff":
            quiet = (airborne_count == 0 and received_count == self._last_received_count
                     and released_count == self._last_released_count)
            self._lip_quiet_since = t if quiet and self._lip_quiet_since is None else (
                None if not quiet else self._lip_quiet_since)
            backed_off = g.tilt_deg(u) <= self._lip_floor_tilt + 0.15 or u <= g.PREPARE_END_U + 1e-6
            settled = self._lip_quiet_since is not None and t - self._lip_quiet_since >= 0.10
            if backed_off or settled:
                self.phase = "lip_settle"
                self._lip_phase_started = t
                self._lip_quiet_since = None
                self._mode_event = "lip_settle"
        if self.phase == "lip_settle":
            quiet = (airborne_count == 0 and received_count == self._last_received_count
                     and released_count == self._last_released_count)
            self._lip_quiet_since = t if quiet and self._lip_quiet_since is None else (
                None if not quiet else self._lip_quiet_since)
            stable = self._lip_quiet_since is not None and t - self._lip_quiet_since >= 0.12
            forced = (self._lip_phase_started is not None and quiet
                      and t - self._lip_phase_started >= 0.50)
            if stable or forced:
                shortfall = self.target_ml - received
                if self._slow_backoff:
                    # Stay at the lip. One more particle only while the cup is still
                    # short of the pass band and there is time to go home.
                    # A 5 degree backoff at 2 deg/s needs about 3.6 s before the
                    # return, so a fine trim does not start inside the last second.
                    band = 0.001 * self.target_ml if self.particle_trim else self.particle_ml
                    reserve = 3.6 if self.particle_trim else 0.85
                    can_trim = (shortfall > band and self._trim_rounds < 4
                                and t + self._time_to_home(u, source_velocity) + reserve < self.deadline)
                    self._lip_quiet_since = None
                    if can_trim:
                        self._trim_rounds += 1
                        self._nudge_released = released_count
                        self._bump_budget = 1
                        self._nudge_span = 6.0
                        self._nudge_tilt_cap = float(g.tilt_deg(u) + self._nudge_span)
                        self.phase = "lip_nudge"
                        self._mode_event = "percent_particle_nudge"
                    else:
                        self._slow_backoff = False
                        self.phase = "retract_78"
                        self._mode_event = "percent_trim_done"
                else:
                    # The fill leaves 0.02-0.08 mL. That is under the 0.2 mL
                    # pulse gate, so the arm used to go home one to four particles short.
                    band = 0.001 * self.target_ml
                    can_trim = (self.particle_trim and band < shortfall <= 0.15 and self._trim_rounds < 4
                                and t + self._time_to_home(u, source_velocity) + 3.6 < self.deadline)
                    self._lip_quiet_since = None
                    if can_trim:
                        self._trim_rounds += 1
                        self._slow_backoff = True
                        self._nudge_released = released_count
                        self._bump_budget = 1
                        self._nudge_span = 6.0
                        self._nudge_tilt_cap = float(g.tilt_deg(u) + self._nudge_span)
                        self.phase = "lip_nudge"
                        self._mode_event = "fine_particle_nudge"
                    else:
                        can_pulse = (self.enable_topup and shortfall > 0.2 and self._topup_count < 4
                                     and t + self._time_to_home(u, source_velocity) + 0.85 < self.deadline)
                        if can_pulse:
                            self._topup_count += 1
                            self._nudge_released = released_count
                            # One shot sized to the measured hole. The 0.45 mL tail model
                            # cuts a 0.25 mL correction before any particle leaves.
                            self._bump_budget = max(1, min(14, int(math.floor((shortfall - 0.06) / self.particle_ml))))
                            self._nudge_tilt_cap = float(g.tilt_deg(u) + self._nudge_span)
                            self.phase = "lip_nudge"
                            self._mode_event = "lip_nudge"
                        else:
                            self.phase = "retract_78"
                            self._mode_event = "lip_done_retract"
        if self.phase == "lip_nudge":
            released_now = released_count - self._nudge_released
            hit_cap = g.tilt_deg(u) >= self._nudge_tilt_cap or u >= self.max_source - 1e-4
            out_of_time = t + self._time_to_home(u, source_velocity) >= self.deadline - 0.05
            if released_now >= self._bump_budget or out_of_time:
                self._lip_quiet_since = None
                if self._slow_backoff and released_now >= self._bump_budget and not out_of_time:
                    self.phase = "lip_backoff"
                    self._lip_floor_tilt = float(g.tilt_deg(u) - 5.0)
                    self._mode_event = "percent_particle_backoff"
                else:
                    self.phase = "retract_78"
                    self._mode_event = "lip_bump_cut"
            elif hit_cap and released_now <= 0 and self._nudge_span < 7.0:
                self._nudge_span = min(7.0, self._nudge_span + 2.0)
                self._nudge_tilt_cap = float(g.tilt_deg(u) + 2.0)
                self._mode_event = "lip_bump_extend"
            elif hit_cap:
                self._lip_quiet_since = None
                self.phase = "retract_78"
                self._mode_event = "lip_bump_cut"
        if self.phase == "topup":
            released_now = released_count - self._topup_start_released
            # Low flow is the 20 mL dribble: its tail is the 0.45 mL floor, and
            # those pours already landed inside ±0.2 mL. The extra 0.30 mL is
            # only for a stream at or above 4 mL/s, where retract made the flow
            # climb and one 40 mL case landed 0.47 mL over. Scale feedback does
            # not use that prediction; the released-particle budget is the cut.
            if self.scale_feedback or self.scale_pulses:
                still_coming = 0.0
            elif self._flow >= 4.0:
                # A 35 mL cut at 6.7 mL/s with a 0.30 mL extra finished 0.228 short,
                # and the following lip nudge released nothing. The 40 mL pour that
                # motivated the extra landed only 0.09 mL over, so the extra is 0.20.
                # fine_percent adds another 0.15 mL so the fill stops short of the
                # one-particle band and the slow trim can add, not remove, liquid.
                still_coming = max(0.45, self._flow * 0.175) + 0.20 + (0.15 if self.fine_percent else 0.0)
            elif self._flow >= 1.5 or released_now >= 6:
                still_coming = max(0.45, max(0.0, self._flow) * 0.175) + (0.15 if self.fine_percent else 0.0)
            else:
                still_coming = 0.0
            predicted = received + inflight + still_coming
            out_of_time = t + self._home_budget() >= self.deadline - 0.05
            at_limit = u >= self.max_source - 1e-4
            budget_hit = released_now >= self._topup_budget
            predicted_hit = (not self.scale_feedback and not self.scale_pulses) and predicted >= self.target_ml
            if budget_hit or predicted_hit or at_limit or out_of_time:
                if self.scale_feedback or self.scale_pulses or out_of_time or at_limit:
                    self.phase = "retract_78"
                    self._mode_event = ("scale_pulse_cut" if self.scale_pulses
                                        else "scale_budget_cut" if self.scale_feedback else "topup_cut")
                else:
                    self.phase = "lip_backoff"
                    self._return_from_fill = True
                    self._bump_budget = 0
                    # Five degrees stops the stream. One and a half degrees
                    # left the mouth in the pour and added about 0.2 mL extra.
                    # The climb back is 6 deg/s so a short hole does not spend
                    # the whole remaining window getting back to the lip.
                    self._nudge_span = 6.0
                    self._lip_floor_tilt = float(g.tilt_deg(u) - 5.0)
                    self._lip_quiet_since = None
                    self._mode_event = "topup_cut_to_lip"
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
                           and t - self._hold_stable_since >= base.ANCHOR_HOLD_QUIET_SECONDS - 1e-9)
            if stable_hold or t - self._hold_started >= base.ANCHOR_HOLD_MAX_SECONDS - 1e-9:
                shortfall = self.target_ml - received
                correction_limit = 6 if self.fine_percent else 4 if self.scale_pulses else 3 if self.scale_feedback else 2
                short_floor = self.particle_ml if self.fine_percent else 0.2
                can_correct = (not self.lip_pulses and not self._return_from_fill and self.enable_topup
                               and self._topup_count < correction_limit and shortfall > short_floor
                               and t + self._home_budget() + 0.35 < self.deadline)
                self._hold_release_reason = "source_and_received_stable_no_airborne" if stable_hold else "hold_maximum_reached"
                self._hold_finished = t
                if can_correct:
                    self._topup_count += 1
                    self._topup_start_released = released_count
                    # Leave a tail margin. A fast tilt releases a gulp larger than the shortfall.
                    if self.fine_percent and (self._topup_count > 1 or shortfall <= 0.35):
                        self._active_topup_tilt = min(self.topup_tilt_deg_s, 2.2)
                        self._slow_backoff = True
                        self._trim_rounds = 0
                        deficit_particles = 1
                    elif self.scale_pulses:
                        # First quiet reading: close a multi-millilitre hole and
                        # leave the margin for the mouth residue. Later readings
                        # add at most a few particles, then the scale is read again.
                        if self._topup_count == 1 and shortfall > self.chunk_margin_ml + 0.40:
                            command_ml = max(self.particle_ml, shortfall - self.chunk_margin_ml)
                            self._active_topup_tilt = self.chunk_tilt_deg_s
                        else:
                            command_ml = max(self.particle_ml, min(0.18, shortfall - 0.06))
                            self._active_topup_tilt = min(self.topup_tilt_deg_s, 2.2)
                        deficit_particles = max(1, int(math.floor(command_ml / self.particle_ml)))
                    elif self.scale_feedback:
                        # The scale is read only after the cup is quiet. Each
                        # addition is capped so one tail cannot spend the whole hole.
                        command_ml = min(1.2, max(self.particle_ml, shortfall - 0.40))
                        deficit_particles = max(1, int(math.floor(command_ml / self.particle_ml)))
                        fast = shortfall > 1.0 and self._topup_count == 1
                        self._active_topup_tilt = self.topup_tilt_deg_s if fast else min(self.topup_tilt_deg_s, 2.2)
                    else:
                        deficit_particles = max(1, int(math.floor((shortfall - 0.25) / self.particle_ml)))
                        fast = shortfall > 1.0 and self._topup_count == 1
                        self._active_topup_tilt = self.topup_tilt_deg_s if fast else min(self.topup_tilt_deg_s, 2.2)
                    self._topup_budget = deficit_particles
                    self.phase = "topup"
                    self._mode_event = "measured_shortfall_topup"
                else:
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
        if (self.phase == "settle" and not self.timed_out and self._stable_frames >= base.STABLE_FRAMES
                and t - self._stable_since >= base.HOME_QUIET_SECONDS - 1e-9 and t <= self.deadline + 1e-9):
            self.phase, self.done = "done", True
            self._completed_at = t
            self._mode_event = "home_quiet_complete"
        if not self.done and t >= self.deadline - 1e-9:
            self.phase, self.timed_out = "timeout_return", True
            self._mode_event = "deadline_timeout_latched"
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
                "motion_continuity": "C2 physical reverse prepare clock"}
        elif self.phase != "prepare":
            returning = self.phase in ("retract_78", "home_return", "settle", "done", "timeout_return",
                                       "lip_backoff")
            angular_derivative = abs(math.degrees(g.angle_derivative(u)))
            if self.phase == "topup":
                rate = min(1.0, self._active_topup_tilt / max(angular_derivative, 1e-9))
            elif self.phase == "lip_nudge":
                nudge_speed = 2.0 if self._slow_backoff else 6.0
                rate = min(1.0, nudge_speed / max(angular_derivative, 1e-9))
            elif self.phase == "lip_backoff" and self._slow_backoff:
                rate = min(1.0, 2.0 / max(angular_derivative, 1e-9))
            else:
                rate = (self.policy.params["return_fraction"] if returning else
                        min(1.0, self.policy.params["dose_tilt_deg_s"] / max(angular_derivative, 1e-9)))
            if returning and home_static:
                rate = 0.0
            if self.phase in ("hold_78", "lip_settle"):
                rate = 0.0
            self._command(t, u, source_velocity, -rate if returning else rate, dt)
        else:
            rate = self.policy.params["prepare_fraction"]
            self._last_command = {"command_goal_source": g.PREPARE_END_U, "command_max_rate": rate,
                "command_acceleration": 0.0, "command_source_velocity": source_velocity,
                "command_next_source_velocity": source_velocity,
                "command_horizon_s": max(0., self._prepare_end - t),
                "motion_continuity": "C2 physical prepare clock"}
        action.update(rate=float(rate), decision=self.phase, return_reason=self._return_reason)
        self.status = {"phase": self.phase, "direct_phase": self.phase, "done": self.done,
            "target_ml": self.target_ml, "received_ml": float(received), "released_ml": float(released),
            "inflight_ml": float(inflight), "flow_ml_s": float(self._flow),
            "source_airborne_particles": airborne_count, "policy_name": POLICY_KIND,
            "policy_params": dict(self._public_params), "policy_action": action,
            "timed_out": self.timed_out, "completed_by_deadline": self.done and not self.timed_out,
            "completed_physical_time_s": self._completed_at, "home_static": home_static,
            "stable_frames": self._stable_frames, "error_ml": float(received - self.target_ml),
            "correction_count": self._topup_count, "correction_enabled": self.enable_topup,
            "deadline_s": self.deadline, "mode_event": self._mode_event,
            "scale_ml": float(received),
            "sensor": ("beaker_scale_after_settled_coarse" if self.scale_pulses
                       else "beaker_scale" if self.scale_feedback else "particle_inflight"),
            "inflight_used_for_control": not self.scale_feedback,
            "correction_uses_scale_only": self.scale_feedback or self.scale_pulses,
            **self._last_command}
        self._last_t, self._last_points = t, points.copy()
        self._last_received_count, self._last_released = received_count, float(released)
        self._last_released_count = released_count
        return dict(self.status)
