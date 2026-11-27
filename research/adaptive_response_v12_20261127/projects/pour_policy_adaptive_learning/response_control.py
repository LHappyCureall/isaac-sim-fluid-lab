"""Live response collection on the unchanged fine-particle PBF environment.

One episode starts at pregrasp and uses the old coarse prefix, then makes one
declared small action from an actually quiet near-onset pose.  Approach water
is accounted separately.  The response includes the local retreat and the
complete cut to 78 degrees, measured only after actual quiet.  Nothing here
writes particle arrays, restores liquid by pose, or predicts a final dose.
"""

import bisect
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PARENT_FILE = ROOT / "projects/pour_policy_training_v10/range_control.py"
PARENT_SHA256 = "7e76bf74c8d266fc7a3eb47eeeba2f6f1ed68880f64835b2114cf030927df729"
if hashlib.sha256(PARENT_FILE.read_bytes()).hexdigest() != PARENT_SHA256:
    raise ValueError("Response collection requires its frozen V10 motion implementation.")
_SPEC = importlib.util.spec_from_file_location("response_frozen_v10_motion", PARENT_FILE)
parent = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = parent
_SPEC.loader.exec_module(parent)

POLICY_KIND = "actual_settled_small_action_response_collection_v1"
PARAMETER_NAMES = parent.PARAMETER_NAMES


class CorrectionPolicy(parent.CorrectionPolicy):
    """Units are degrees, physical seconds, and degrees per physical second."""

    name = POLICY_KIND

    def __init__(self, coarse, deadline_s=24., approach_offset_deg=-.5,
                 approach_tilt_deg_s=6., action_delta_deg=.03,
                 action_tilt_deg_s=.6, hold_seconds=.1,
                 backoff_delta_deg=.5, backoff_tilt_deg_s=4.,
                 anchor_return_tilt_deg_s=6., quiet_seconds=.25,
                 maximum_measurement_hold_s=1.):
        super().__init__(coarse, deadline_s=deadline_s, enable_topup=False,
                         quiet_seconds=quiet_seconds,
                         approach_tilt_deg_s=approach_tilt_deg_s,
                         return_tilt_deg_s=anchor_return_tilt_deg_s)
        values = dict(approach_offset_deg=approach_offset_deg,
                      approach_tilt_deg_s=approach_tilt_deg_s,
                      action_delta_deg=action_delta_deg,
                      action_tilt_deg_s=action_tilt_deg_s,
                      hold_seconds=hold_seconds,
                      backoff_delta_deg=backoff_delta_deg,
                      backoff_tilt_deg_s=backoff_tilt_deg_s,
                      anchor_return_tilt_deg_s=anchor_return_tilt_deg_s,
                      maximum_measurement_hold_s=maximum_measurement_hold_s)
        bounds = dict(approach_offset_deg=(-6., 6.), approach_tilt_deg_s=(1., 6.),
                      action_delta_deg=(-.5, .5), action_tilt_deg_s=(.1, 4.),
                      hold_seconds=(0., .4), backoff_delta_deg=(.025, 2.),
                      backoff_tilt_deg_s=(.1, 6.), anchor_return_tilt_deg_s=(1., 6.),
                      maximum_measurement_hold_s=(.3, 2.))
        for key, value in values.items():
            if (isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value))
                    or not bounds[key][0] <= float(value) <= bounds[key][1]):
                raise ValueError("Response experiment parameter outside finite bounds: " + key)
            setattr(self, key, float(value))
        if self.maximum_measurement_hold_s < self.quiet_seconds:
            raise ValueError("Maximum hold cannot be shorter than actual quiet requirement.")
        self.params = {**self.params, **{key: getattr(self, key) for key in values},
                       "experiment_kind": POLICY_KIND,
                       "small_action_final_tail_prediction_used": False,
                       "response_cutoff_tilt_deg": 78.,
                       "response_start_requires_actual_quiet": True,
                       "one_declared_small_action_per_episode": True}


class TargetPourController(parent.TargetPourController):
    """Committed, joint-limited trajectories plus read-only observations."""

    def __init__(self, geometry, target_ml=30., spacing=.0027, policy=None):
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Response collection requires its own CorrectionPolicy.")
        super().__init__(geometry, target_ml, spacing, policy)

    def reset(self, target_ml=None):
        result = super().reset(target_ml)
        self.response_events = []
        self._response_intercepted = False
        self._response_action_start = None
        self._response_approach_start = None
        self._response_wait_started = None
        self._response_final_reason = None
        self._response_last_update = None
        self._response_source_motion = {}
        self.status["policy_name"] = POLICY_KIND
        self.status["response_events"] = []
        return dict(self.status)

    def _start_home(self, t, u):
        if not self._response_intercepted:
            self._response_intercepted = True
            self.phase = "response_anchor_ready"
            return
        return super()._start_home(t, u)

    def _observation(self, t, status):
        source_count = int(np.count_nonzero(self._fixed_labels == 1))
        released_count = int(self._last_released_count)
        return {"physical_time_s": float(t),
                "received_ml": float(status["received_ml"]),
                "tube_remaining_ml": (source_count - released_count) * self.particle_ml,
                "tilt_deg": float(status["tilt_deg"]),
                "source_airborne_particles": int(status["source_airborne_particles"]),
                "net_outside_particles": released_count,
                "inside_to_outside_events": int(status["inside_to_outside_events"]),
                "new_inside_to_outside_events": int(status["new_inside_to_outside_events"]),
                "ever_exited_particle_count": int(status["ever_exited_particle_count"]),
                "quiet_measured_seconds": float(status["quiet_measured_seconds"]),
                "settled_measurement": bool(status["settled_measurement"]),
                "actual_u_velocity": float(status["actual_u_velocity"]),
                "actual_u_acceleration": float(status["actual_u_acceleration"]),
                "tube_pose_measurement_basis": status["tube_pose_measurement_basis"],
                "remaining_deadline_s": self.deadline - float(t),
                **self._response_source_motion}

    def _observe_source_motion(self, points, velocities, observed_pose, status, basis):
        """Retain actual internal-motion context without changing liquid state.

        Quiet membership is a dose-measurement condition, not a declaration of
        zero internal velocity.  The exact inside identities are those just
        classified by the parent's actual-full-R observation.
        """
        inside = self._previous_inside
        selected_points = points[inside]
        selected_velocity = velocities[inside]
        count = int(np.count_nonzero(inside))
        features = {"source_inside_particle_count": count,
                    "source_velocity_measurement_basis": basis,
                    "dose_quiet_does_not_imply_zero_internal_motion": True}
        if count:
            center = np.mean(selected_points, axis=0)
            speed_squared = np.sum(selected_velocity * selected_velocity, axis=1)
            origin, _ = self.geometry.tube_pose(float(status["u"]))
            rotation = self.geometry.tube_rotation(float(status["u"]))
            if observed_pose is not None:
                origin = observed_pose[:3]
                x, y, z, w = observed_pose[3:]
                rotation = np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
            features.update(source_inside_centroid_world_m=center.tolist(),
                            source_inside_centroid_local_m=((center - origin) @ rotation).tolist(),
                            source_inside_mean_velocity_world_m_s=np.mean(selected_velocity, axis=0).tolist(),
                            source_inside_velocity_rms_m_s=float(np.sqrt(np.mean(speed_squared))),
                            source_inside_velocity_max_m_s=float(np.sqrt(np.max(speed_squared))))
        else:
            features.update(source_inside_centroid_world_m=None,
                            source_inside_centroid_local_m=None,
                            source_inside_mean_velocity_world_m_s=None,
                            source_inside_velocity_rms_m_s=None,
                            source_inside_velocity_max_m_s=None)
        self._response_source_motion = features

    def _event(self, kind, t, status, **details):
        event = {"event_id": len(self.response_events), "type": kind,
                 "physical_time_s": float(t), "observation": self._observation(t, status),
                 **details}
        self.response_events.append(event)
        return event

    def _action(self):
        p = self.profile
        return {"delta_deg": p.action_delta_deg,
                "tilt_speed_deg_s": p.action_tilt_deg_s,
                "hold_s": p.hold_seconds,
                "retract_deg": p.backoff_delta_deg,
                "retract_speed_deg_s": p.backoff_tilt_deg_s,
                "cutoff_tilt_deg": 78.,
                "cutoff_speed_deg_s": p.anchor_return_tilt_deg_s}

    def _planned_budget(self, u, approach_angle):
        p, g = self.profile, self.geometry
        angles = (approach_angle, approach_angle + p.action_delta_deg,
                  max(78., approach_angle + p.action_delta_deg - p.backoff_delta_deg), 78.)
        rates = (p.approach_tilt_deg_s, p.action_tilt_deg_s,
                 p.backoff_tilt_deg_s, p.anchor_return_tilt_deg_s)
        total = 0.
        for angle, rate in zip(angles, rates):
            goal = float(g.u_for_tilt(angle))
            duration, _ = self._duration_for_step(u, goal, rate)
            total += duration
            u = goal
        return (total + p.hold_seconds + 3 * p.maximum_measurement_hold_s
                + self._home_budget() + .15)

    def _go_cutoff(self, t, u, status, reason):
        self._response_final_reason = reason
        self._event("response_cutoff_start", t, status, reason=reason,
                    action=self._action(), intended_cutoff_tilt_deg=78.)
        self._start_step(t, u, 78., self.profile.anchor_return_tilt_deg_s,
                         "response_cutoff_motion")

    def _finish_response(self, t, u, status, settled):
        if self._response_action_start is not None:
            pre = self._response_action_start
            post = self._observation(t, status)
            outcome = {"settled_delta_ml": post["received_ml"] - pre["observation"]["received_ml"],
                       "duration_s": float(t) - pre["physical_time_s"]}
            self._event("response_action_complete" if settled else "response_action_unsettled",
                        t, status, action=self._action(), settled=bool(settled),
                        pre_event_id=pre["event_id"],
                        pre_physical_time_s=pre["physical_time_s"],
                        post_physical_time_s=float(t), pre_observation=pre["observation"],
                        post_observation=post, outcome=outcome,
                        completion_reason=self._response_final_reason,
                        final_tilt_anchor_deg=78.,
                        measurement_basis="actual_received_change_after_complete_cut_and_observed_quiet")
        else:
            self._event("response_not_executed", t, status,
                        reason=self._response_final_reason, settled=bool(settled))
        super()._start_home(t, u)

    def update(self, t, positions, labels, velocities=None):
        # Duplicate observational reads cannot create additional actions/events.
        if self._response_last_update is not None and abs(float(t) - self._response_last_update) < 1e-9:
            return dict(self.status)
        points = np.asarray(positions, dtype=float)
        observed_pose = (None if self._observed_tube_pose is None
                         else self._observed_tube_pose.copy())
        if velocities is None:
            actual_velocity = ((points - self._last_points) / (float(t) - self._last_t)
                               if self._last_points is not None else np.zeros_like(points))
            velocity_basis = "observed_position_finite_difference" if self._last_points is not None else "initial_unavailable_zero_fixture"
        else:
            actual_velocity = np.asarray(velocities, dtype=float)
            velocity_basis = "provided_actual_particle_qd"
        status = super().update(t, positions, labels, velocities)
        self._observe_source_motion(points, actual_velocity, observed_pose, status, velocity_basis)
        t = float(t)
        u, rate, _, _ = self.motion_state(t)
        p, g = self.profile, self.geometry
        if not self.timed_out:
            if self.phase == "response_anchor_ready":
                approach = float(self._coarse_cut_angle + p.approach_offset_deg)
                # Invalid experiments are recorded; never silently clamp an action.
                if not 78. <= approach <= 100. or not 78. <= approach + p.action_delta_deg <= 100.:
                    self._event("response_not_executed", t, status,
                                reason="declared_angle_outside_supported_path", proposed_approach_deg=approach)
                    super()._start_home(t, u)
                else:
                    budget = self._planned_budget(u, approach)
                    if t + budget > self.deadline:
                        self._event("response_not_executed", t, status,
                                    reason="declared_action_cannot_fit_full_home_deadline",
                                    conservative_remaining_budget_s=budget)
                        super()._start_home(t, u)
                    else:
                        self._response_approach_start = self._event("response_approach_start", t, status,
                            prior_observed_coarse_cut_angle_deg=self._coarse_cut_angle,
                            approach_offset_deg=p.approach_offset_deg,
                            approach_tilt_deg=approach, approach_speed_deg_s=p.approach_tilt_deg_s,
                            declared_action=self._action(), conservative_remaining_budget_s=budget)
                        self._start_step(t, u, approach, p.approach_tilt_deg_s,
                                         "response_approach_motion")
            elif self.phase == "response_approach_motion" and t >= self._pending_end - 1e-9:
                self._hold(t, u, "response_approach_quiet")
                self._response_wait_started = t
                self._event("response_approach_motion_end", t, status)
            elif self.phase == "response_approach_quiet":
                if status["settled_measurement"]:
                    self._event("response_approach_complete", t, status,
                                declared_action=self._action(),
                                approach_received_delta_ml=(status["received_ml"]
                                    - self._response_approach_start["observation"]["received_ml"]))
                    self._response_action_start = self._event("response_action_start", t, status,
                                                              action=self._action())
                    self._start_step(t, u, float(g.tilt_deg(u)) + p.action_delta_deg,
                                     p.action_tilt_deg_s, "response_action_motion")
                elif t - self._response_wait_started >= p.maximum_measurement_hold_s:
                    self._go_cutoff(t, u, status, "approach_did_not_reach_actual_quiet")
            elif self.phase == "response_action_motion" and t >= self._pending_end - 1e-9:
                self._hold(t, u, "response_action_hold")
                self._response_wait_started = t
                self._event("response_action_motion_end", t, status, action=self._action())
            elif self.phase == "response_action_hold" and t - self._response_wait_started >= p.hold_seconds - 1e-9:
                target = max(78., float(g.tilt_deg(u)) - p.backoff_delta_deg)
                self._event("response_local_retreat_start", t, status,
                            actual_retract_deg=float(g.tilt_deg(u)) - target)
                self._start_step(t, u, target, p.backoff_tilt_deg_s,
                                 "response_local_retreat_motion")
            elif self.phase == "response_local_retreat_motion" and t >= self._pending_end - 1e-9:
                self._hold(t, u, "response_local_retreat_quiet")
                self._response_wait_started = t
                self._event("response_local_retreat_motion_end", t, status)
            elif self.phase == "response_local_retreat_quiet":
                if status["settled_measurement"] or t - self._response_wait_started >= p.maximum_measurement_hold_s:
                    self._event("response_local_retreat_observed", t, status,
                                actual_quiet=bool(status["settled_measurement"]))
                    self._go_cutoff(t, u, status,
                                    "declared_action_finished_cut_and_measure")
            elif self.phase == "response_cutoff_motion" and t >= self._pending_end - 1e-9:
                self._hold(t, u, "response_cutoff_quiet")
                self._response_wait_started = t
                self._event("response_cutoff_motion_end", t, status)
            elif self.phase == "response_cutoff_quiet":
                if status["settled_measurement"] or t - self._response_wait_started >= p.maximum_measurement_hold_s:
                    self._finish_response(t, u, status, bool(status["settled_measurement"]))
        elif not any(e["type"] == "response_deadline_failure" for e in self.response_events):
            self._event("response_deadline_failure", t, status,
                        action=self._action(), action_was_started=self._response_action_start is not None)
        status.update(phase=self.phase, direct_phase=self.phase,
                      policy_name=POLICY_KIND, policy_params=dict(p.params),
                      mode_event=(self.response_events[-1]["type"] if self.response_events else self._last_event),
                      response_events=list(self.response_events),
                      response_motion_segment=list(self._segments[max(0, bisect.bisect_right(self._times, t) - 1)]),
                      response_collection_only=True,
                      response_model_used=False,
                      small_action_tail_prediction_used=False,
                      response_action_completed=any(e["type"] == "response_action_complete" for e in self.response_events),
                      **self._last_command)
        status["policy_action"] = {**status["policy_action"], "decision": self.phase,
                                   "small_action_tail_prediction_used": False}
        self.status = status
        self._response_last_update = t
        return dict(status)
