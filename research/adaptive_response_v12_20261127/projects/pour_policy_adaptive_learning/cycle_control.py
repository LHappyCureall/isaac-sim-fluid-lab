"""Whole anchor-to-anchor response actions, including approach drainage.

This is a new controller.  Its frozen parent remains unchanged.  Each cycle
starts at an actually quiet 78-degree pose and includes approach, any executed
micro action, local retreat and complete cutoff.  Actual settled measurements
alone decide whether the requested dose is complete.  A model can propose a
supported cycle; it cannot declare task success or modify the liquid.
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT_FILE = HERE / "response_control.py"
PARENT_SHA256 = "875b8629d04fe17a8f67ec6287cc60d4644ffa562baa1682ac3f1a01a78730a6"
if hashlib.sha256(PARENT_FILE.read_bytes()).hexdigest() != PARENT_SHA256:
    raise ValueError("Whole-cycle control requires its frozen response implementation.")
_SPEC = importlib.util.spec_from_file_location("cycle_frozen_response_motion", PARENT_FILE)
response = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = response
_SPEC.loader.exec_module(response)

POLICY_KIND = "actual_settled_whole_cycle_response_control_v1"
PARAMETER_NAMES = response.PARAMETER_NAMES
ACTION_KEYS = ("approach_delta_deg", "approach_speed_deg_s", "delta_deg", "tilt_speed_deg_s",
               "hold_s", "retract_deg", "retract_speed_deg_s", "cutoff_tilt_deg",
               "cutoff_speed_deg_s", "quiet_seconds", "maximum_measurement_hold_s")


def actual_dose_in_band(target_ml, received_ml):
    tolerance = .001 * float(target_ml)
    return float(target_ml) - tolerance <= float(received_ml) <= float(target_ml) + tolerance


class CorrectionPolicy(response.CorrectionPolicy):
    name = POLICY_KIND

    def __init__(self, coarse, deadline_s=24., approach_tilt_deg=None,
                 approach_offset_deg=-8., approach_tilt_deg_s=6., action_delta_deg=.03,
                 action_tilt_deg_s=.6, hold_seconds=.1, backoff_delta_deg=.5,
                 backoff_tilt_deg_s=4., anchor_return_tilt_deg_s=6., quiet_seconds=.25,
                 maximum_measurement_hold_s=1., maximum_cycles=1,
                 model_checkpoint=None, model_sha256=None, model_action_menu=None):
        if (isinstance(approach_offset_deg, (bool, np.bool_))
                or not math.isfinite(float(approach_offset_deg))
                or not -10. <= float(approach_offset_deg) <= 0.):
            raise ValueError("Relative collection approach offset must be in[-10,0]degrees.")
        if approach_tilt_deg is not None and (isinstance(approach_tilt_deg, (bool, np.bool_))
                or not math.isfinite(float(approach_tilt_deg))
                or not 78. <= float(approach_tilt_deg) <= 100.):
            raise ValueError("Absolute collection approach must be in[78,100]degrees.")
        if isinstance(maximum_cycles, bool) or maximum_cycles not in (1, 2):
            raise ValueError("Only one or two fully measured cycles are supported.")
        # The parent validates the shared motion parameters.  Its older offset
        # range is constructor compatibility only: the actual offset below is
        # separately validated, recorded, and never silently motion-clamped.
        super().__init__(coarse, deadline_s=deadline_s, approach_offset_deg=0.,
                         approach_tilt_deg_s=approach_tilt_deg_s,
                         action_delta_deg=action_delta_deg, action_tilt_deg_s=action_tilt_deg_s,
                         hold_seconds=hold_seconds, backoff_delta_deg=backoff_delta_deg,
                         backoff_tilt_deg_s=backoff_tilt_deg_s,
                         anchor_return_tilt_deg_s=anchor_return_tilt_deg_s,
                         quiet_seconds=quiet_seconds,
                         maximum_measurement_hold_s=maximum_measurement_hold_s)
        self.approach_offset_deg = float(approach_offset_deg)
        self.approach_tilt_deg = None if approach_tilt_deg is None else float(approach_tilt_deg)
        self.maximum_cycles = int(maximum_cycles)
        self.model, self.model_module, self.model_checkpoint_sha256 = None, None, None
        self.model_action_menu = None
        if (model_checkpoint is None) != (model_sha256 is None):
            raise ValueError("A model checkpoint requires an explicit matching raw SHA256.")
        if model_checkpoint is not None:
            checkpoint = Path(model_checkpoint).expanduser().resolve()
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != model_sha256:
                raise ValueError("Response model checkpoint does not match its frozen raw SHA.")
            module_file = HERE / "cycle_response_model.py"
            spec = importlib.util.spec_from_file_location("cycle_runtime_response_model", module_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.model = module.load_response_model(checkpoint, expected_raw_sha256=digest)
            self.model_module = module
            self.model_checkpoint_sha256 = digest
            menu = self.model.get("action_menu") if model_action_menu is None else model_action_menu
            if not isinstance(menu, list) or not menu:
                raise ValueError("A learned cycle controller requires its frozen empirical action menu.")
            self.model_action_menu = [dict(action) for action in menu]
        elif model_action_menu is not None:
            raise ValueError("An action menu alone is not a trained or calibrated model.")
        self.params = {**self.params, "experiment_kind": POLICY_KIND,
                       "approach_offset_deg": self.approach_offset_deg,
                       "approach_tilt_deg": self.approach_tilt_deg,
                       "approach_basis": "explicit_absolute_angle" if approach_tilt_deg is not None else "previous_coarse_cut_plus_offset",
                       "maximum_cycles": self.maximum_cycles,
                       "one_declared_small_action_per_episode": self.maximum_cycles == 1,
                       "whole_cycle_includes_approach_drainage": True,
                       "model_checkpoint_sha256": self.model_checkpoint_sha256,
                       "model_action_menu": self.model_action_menu,
                       "actual_closed_interval_dose_decides_home": True}


class TargetPourController(response.TargetPourController):
    def __init__(self, geometry, target_ml=30., spacing=.0027, policy=None):
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Whole-cycle control requires its own CorrectionPolicy.")
        super().__init__(geometry, target_ml, spacing, policy)

    def reset(self, target_ml=None):
        result = super().reset(target_ml)
        self._cycle_count = 0
        self._cycle_start = None
        self._cycle_action = None
        self._cycle_intercepted = False
        self._cycle_model_decisions = []
        self._cycle_observed_peak_spill = None
        self._cycle_terminal_reason = None
        self.status["policy_name"] = POLICY_KIND
        return dict(self.status)

    def set_observed_peak_spill_particles(self, count):
        """Optional read-only metrics hook for the independent environment."""
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("Observed all-label peak spill must be a nonnegative integer.")
        self._cycle_observed_peak_spill = max(int(count), self._cycle_observed_peak_spill or 0)

    def _start_home(self, t, u):
        if not self._cycle_intercepted:
            self._cycle_intercepted = True
            self.phase = "response_cycle_ready"
            return
        return response.parent.TargetPourController._start_home(self, t, u)

    def _real_home(self, t, u, status, reason):
        self._cycle_terminal_reason = reason
        self._event("response_cycle_home_decision", t, status, reason=reason,
                    actual_dose_in_closed_band=actual_dose_in_band(self.target_ml, status["received_ml"]))
        response.parent.TargetPourController._start_home(self, t, u)

    def _observation(self, t, status):
        result = super()._observation(t, status)
        centroid = result.get("source_inside_centroid_local_m")
        result["source_inside_centroid_local_z_m"] = None if centroid is None else float(centroid[2])
        result["reference_angle_deg"] = float(self._coarse_cut_angle)
        return result

    def _fixed_cycle_action(self, status):
        p = self.profile
        approach = (p.approach_tilt_deg if p.approach_tilt_deg is not None
                    else self._coarse_cut_angle + p.approach_offset_deg)
        return {"approach_delta_deg": float(approach - status["tilt_deg"]),
                "approach_speed_deg_s": p.approach_tilt_deg_s,
                "delta_deg": p.action_delta_deg, "tilt_speed_deg_s": p.action_tilt_deg_s,
                "hold_s": p.hold_seconds, "retract_deg": p.backoff_delta_deg,
                "retract_speed_deg_s": p.backoff_tilt_deg_s,
                "cutoff_tilt_deg": 78., "cutoff_speed_deg_s": p.anchor_return_tilt_deg_s,
                "quiet_seconds": p.quiet_seconds,
                "maximum_measurement_hold_s": p.maximum_measurement_hold_s}

    def _action(self):
        return super()._action() if self._cycle_action is None else dict(self._cycle_action)

    def _choose_cycle_action(self, t, status):
        p = self.profile
        if p.model is None:
            return self._fixed_cycle_action(status)
        if self._cycle_observed_peak_spill is None:
            self._cycle_model_decisions.append({"physical_time_s": t, "mode": "abstain",
                                               "reason": "independent_all_label_peak_spill_not_observed"})
            return None
        observation = self._observation(t, status)
        state = {key: observation[key] for key in
                 ("received_ml", "tube_remaining_ml", "tilt_deg",
                  "source_inside_velocity_rms_m_s", "source_inside_centroid_local_z_m")}
        decision = p.model_module.choose_action(p.model, state, p.model_action_menu,
            target_ml=self.target_ml, remaining_time_s=self.deadline - t,
            home_reserve_s=self._home_budget(),
            current_peak_spill_particles=self._cycle_observed_peak_spill)
        self._cycle_model_decisions.append({"physical_time_s": t, **decision})
        if decision.get("mode") != "execute_then_actual_settled_remeasure":
            return None
        return dict(decision["action"])

    def _apply_cycle_action(self, action):
        if set(action) != set(ACTION_KEYS) or any(isinstance(v, bool) or not math.isfinite(float(v))
                                                for v in action.values()):
            raise ValueError("Cycle action must contain exactly eleven finite physical parameters.")
        bounds = {"approach_delta_deg": (0., 22.), "approach_speed_deg_s": (1., 6.),
                  "delta_deg": (-.5, .5), "tilt_speed_deg_s": (.1, 4.), "hold_s": (0., .4),
                  "retract_deg": (.025, 2.), "retract_speed_deg_s": (.1, 6.),
                  "cutoff_tilt_deg": (78., 78.), "cutoff_speed_deg_s": (1., 6.),
                  "quiet_seconds": (.2, .6), "maximum_measurement_hold_s": (.3, 2.)}
        for key, (low, high) in bounds.items():
            if not low <= action[key] <= high:
                raise ValueError("Chosen cycle action outside physical bounds: " + key)
        if action["maximum_measurement_hold_s"] < action["quiet_seconds"]:
            raise ValueError("Chosen cycle has inconsistent quiet/hold parameters.")
        mapping = {"approach_speed_deg_s": "approach_tilt_deg_s", "delta_deg": "action_delta_deg",
                   "tilt_speed_deg_s": "action_tilt_deg_s", "hold_s": "hold_seconds",
                   "retract_deg": "backoff_delta_deg", "retract_speed_deg_s": "backoff_tilt_deg_s",
                   "cutoff_speed_deg_s": "anchor_return_tilt_deg_s",
                   "quiet_seconds": "quiet_seconds",
                   "maximum_measurement_hold_s": "maximum_measurement_hold_s"}
        # Runtime action parameters are explicit in every cycle event.  The
        # original policy declaration remains in params as the frozen contract.
        for source, destination in mapping.items():
            setattr(self.profile, destination, float(action[source]))
        self._cycle_action = {key: float(action[key]) for key in ACTION_KEYS}

    def _finish_response(self, t, u, status, settled):
        if self._response_action_start is not None:
            pre = self._response_action_start
            post = self._observation(t, status)
            self._event("response_action_complete" if settled else "response_action_unsettled",
                t, status, action=self._action(), settled=bool(settled), pre_event_id=pre["event_id"],
                pre_physical_time_s=pre["physical_time_s"], post_physical_time_s=float(t),
                pre_observation=pre["observation"], post_observation=post,
                outcome={"settled_delta_ml": post["received_ml"] - pre["observation"]["received_ml"],
                         "duration_s": float(t) - pre["physical_time_s"]})
        pre = self._cycle_start
        post = self._observation(t, status)
        executed = self._response_action_start is not None
        self._event("response_cycle_complete" if settled else "response_cycle_unsettled", t, status,
            action=self._action(), settled=bool(settled), pre_event_id=pre["event_id"],
            pre_physical_time_s=pre["physical_time_s"], post_physical_time_s=float(t),
            pre_observation=pre["observation"], post_observation=post,
            outcome={"settled_delta_ml": post["received_ml"] - pre["observation"]["received_ml"],
                     "duration_s": float(t) - pre["physical_time_s"],
                     "executed_micro": executed}, executed_micro=executed,
            action_execution_complete=bool(settled and executed),
            completion_reason=self._response_final_reason,
            approach_received_delta_ml=(None if not executed else
                self._response_action_start["observation"]["received_ml"] - pre["observation"]["received_ml"]),
            measurement_basis="actual_anchor_to_anchor_received_change_including_approach_and_cutoff")
        self.phase = "response_cycle_ready"
        if not settled:
            self._real_home(t, u, status, "cycle_never_reached_actual_quiet")

    def update(self, t, positions, labels, velocities=None):
        # Parent guards duplicate observations.  Do not independently replay a
        # cycle decision after that early-return path.
        repeated = self._response_last_update is not None and abs(float(t) - self._response_last_update) < 1e-9
        status = super().update(t, positions, labels, velocities)
        if repeated:
            return dict(self.status)
        t = float(t)
        u = self.source_time(t)
        if self.phase == "response_cycle_ready" and not self.timed_out:
            received = float(status["received_ml"])
            if actual_dose_in_band(self.target_ml, received):
                self._real_home(t, u, status, "actual_settled_dose_in_closed_interval")
            elif received > self.target_ml + .001 * self.target_ml:
                self._real_home(t, u, status, "actual_overdose_cannot_be_removed")
            elif self._cycle_count >= self.profile.maximum_cycles:
                self._real_home(t, u, status, "declared_cycle_budget_finished")
            elif not status["settled_measurement"]:
                self._real_home(t, u, status, "cycle_start_requires_actual_quiet")
            else:
                action = self._choose_cycle_action(t, status)
                if action is None:
                    self._real_home(t, u, status, "model_abstained_or_requested_home")
                else:
                    self._apply_cycle_action(action)
                    approach = float(status["tilt_deg"] + action["approach_delta_deg"])
                    if not 78. <= approach <= 100. or not 78. <= approach + action["delta_deg"] <= 100.:
                        self._real_home(t, u, status, "declared_cycle_angle_outside_path")
                    else:
                        budget = self._planned_budget(u, approach)
                        if t + budget > self.deadline:
                            self._real_home(t, u, status, "whole_cycle_cannot_fit_full_home_deadline")
                        else:
                            self._cycle_count += 1
                            self._response_action_start = None
                            self._response_final_reason = None
                            self._cycle_start = self._event("response_cycle_start", t, status,
                                cycle_index=self._cycle_count, action=self._action(),
                                reference_angle_deg=self._coarse_cut_angle,
                                approach_tilt_deg=approach,
                                approach_offset_from_reference_deg=approach - self._coarse_cut_angle,
                                conservative_remaining_budget_s=budget,
                                model_checkpoint_sha256=self.profile.model_checkpoint_sha256)
                            self._response_approach_start = self._event("response_approach_start", t, status,
                                prior_observed_coarse_cut_angle_deg=self._coarse_cut_angle,
                                approach_offset_deg=approach - self._coarse_cut_angle,
                                approach_tilt_deg=approach, approach_speed_deg_s=action["approach_speed_deg_s"],
                                declared_action=self._action(), conservative_remaining_budget_s=budget)
                            self._start_step(t, u, approach, self.profile.approach_tilt_deg_s,
                                             "response_approach_motion")
        status.update(phase=self.phase, direct_phase=self.phase, policy_name=POLICY_KIND,
                      response_events=list(self.response_events), response_cycle_count=self._cycle_count,
                      response_cycle_terminal_reason=self._cycle_terminal_reason,
                      response_cycle_active_action=None if self._cycle_action is None else dict(self._cycle_action),
                      response_cycle_model_decisions=list(self._cycle_model_decisions),
                      response_cycle_observed_peak_spill_particles=self._cycle_observed_peak_spill,
                      response_model_used=bool(self._cycle_model_decisions),
                      response_collection_only=self.profile.model is None,
                      response_motion_segment=list(self._segments[-1]),
                      **self._last_command)
        status["policy_action"] = {**status["policy_action"], "decision": self.phase}
        self.status = status
        return dict(status)
