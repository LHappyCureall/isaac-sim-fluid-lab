"""Fixed20mL measured feedback with at most one conditional second cycle.

The first approach is84.0degrees. A second84.1degree approach is allowed only
after actual quiet, a measured deficit greater than0.5mL, a current all-label
spill count <=1 and a conservative full-home time check. No angle is trained
or selected in this new study, and no liquid state or old source is changed.
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PARENT_FILE = ROOT / "projects/pour_policy_adaptive_learning/cycle_control.py"
PARENT_SHA256 = "927d2b7600ec1d0d40a34296659c29bb4727e73969a1d9810648347972a79daa"
if hashlib.sha256(PARENT_FILE.read_bytes()).hexdigest() != PARENT_SHA256:
    raise ValueError("Retry20 requires the frozen measured whole-cycle controller")
_SPEC = importlib.util.spec_from_file_location("retry20_frozen_cycle_controller", PARENT_FILE)
cycle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cycle
_SPEC.loader.exec_module(cycle)

POLICY_KIND = "fixed20_conditional_settled_retry_control_v1"
PARAMETER_NAMES = cycle.PARAMETER_NAMES
FIXED_COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
                "return_fraction": 1., "tail_lead_s": .26,
                "stop_bias_ml": 10., "max_dose_u_accel": 6.}
TARGET_ML, TOLERANCE_ML = 20., .5
FIRST_APPROACH_DEG, RETRY_APPROACH_DEG = 84., 84.1
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _coarse(value):
    if value is None:
        return dict(FIXED_COARSE)
    if (not isinstance(value, dict) or set(value) != set(FIXED_COARSE)
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   or value[key] != fixed for key, fixed in FIXED_COARSE.items())):
        raise ValueError("Retry20 fixes all six original coarse parameters")
    return dict(FIXED_COARSE)


def in_task_band(received_ml):
    if type(received_ml) not in (int, float) or not math.isfinite(received_ml):
        raise ValueError("Retry20 requires finite actual received water")
    return TARGET_ML - TOLERANCE_ML <= received_ml <= TARGET_ML + TOLERANCE_ML


class CorrectionPolicy(cycle.CorrectionPolicy):
    name = POLICY_KIND

    def __init__(self, coarse=None, enable_retry=True, deadline_s=24.):
        if type(enable_retry) is not bool:
            raise ValueError("enable_retry must be an explicit boolean")
        if type(deadline_s) not in (int, float) or deadline_s != 24.:
            raise ValueError("Retry20 keeps the24s original physical clock")
        self.enable_retry = enable_retry
        super().__init__(_coarse(coarse), deadline_s=24.,
            approach_tilt_deg=FIRST_APPROACH_DEG, approach_tilt_deg_s=6.,
            action_delta_deg=.03, action_tilt_deg_s=.6, hold_seconds=.1,
            backoff_delta_deg=.5, backoff_tilt_deg_s=4.,
            anchor_return_tilt_deg_s=6., quiet_seconds=.25,
            maximum_measurement_hold_s=1., maximum_cycles=2 if enable_retry else 1)
        self.params.update(experiment_kind=POLICY_KIND, enable_retry=enable_retry,
            enable_correction=True, first_approach_tilt_deg=FIRST_APPROACH_DEG,
            retry_approach_tilt_deg=RETRY_APPROACH_DEG,
            retry_actual_deficit_threshold_ml=TOLERANCE_ML,
            retry_maximum_all_label_peak_spill_particles=1,
            retry_requires_actual_quiet=True, retry_requires_full_home_budget=True,
            retry_task_target_ml=TARGET_ML, retry_absolute_tolerance_ml=TOLERANCE_ML,
            actual_measurement_target_unchanged=True, coarse_control_target_offset_ml=0.,
            all_motion_speeds_fixed=True, no_angle_training_or_selection=True,
            settled_actual_volume_decides_home=True, original_strict_band_unchanged=True,
            unchanged_precision_v2_controller=False)


class TargetPourController(cycle.TargetPourController):
    def __init__(self, geometry, target_ml=20., spacing=.0027, policy=None):
        if type(target_ml) not in (int, float) or target_ml != TARGET_ML:
            raise ValueError("Retry20 has exactly one actual20mL goal")
        if type(spacing) not in (int, float) or spacing != .0027:
            raise ValueError("Retry20 keeps the2.7mm actual particles")
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Retry20 requires its declared CorrectionPolicy")
        super().__init__(geometry, target_ml, spacing, policy)

    def reset(self, target_ml=None):
        if target_ml is not None and (type(target_ml) not in (int, float) or target_ml != TARGET_ML):
            raise ValueError("Reset may not change the actual20mL goal")
        super().reset(target_ml)
        self._retry_anchor_requested_at_s = None
        self._retry_in_band_home = False
        self._retry_blocked_reason = None
        self._retry_decisions = []
        self._retry_selected_actual_approach_deg = None
        self.prefix_anchor_observation = None
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params), **self._metadata())
        return dict(self.status)

    def _metadata(self):
        actual_angles = [float(event["approach_tilt_deg"]) for event in self.response_events
                         if event["type"] == "response_cycle_start"]
        return {"actual_target_ml": TARGET_ML, "control_target_ml": TARGET_ML,
            "control_target_offset_ml": 0., "prefix_target_offset_ml": 0.,
            "retry_controller_source_sha256": _SOURCE_SHA256,
            "enable_retry": self.profile.enable_retry, "retry_enabled": self.profile.enable_retry,
            "retry_first_approach_tilt_deg": FIRST_APPROACH_DEG,
            "retry_second_approach_tilt_deg": RETRY_APPROACH_DEG,
            "retry_maximum_cycles": self.profile.maximum_cycles,
            "retry_selected_approach_tilt_deg": self._retry_selected_actual_approach_deg,
            "retry_actual_approach_tilt_deg": actual_angles[-1] if actual_angles else None,
            "retry_actual_cycle_approach_tilt_degrees": actual_angles,
            "retry_absolute_tolerance_ml": TOLERANCE_ML,
            "actual_measurement_target_unchanged": True,
            "coarse_motion_parameters_fixed": True, "all_motion_speeds_fixed": True}

    def _start_home(self, t, u):
        if self._retry_anchor_requested_at_s is None:
            self._retry_anchor_requested_at_s = float(t)
        return super()._start_home(t, u)

    def _choose_cycle_action(self, t, status):
        received = float(status["received_ml"])
        quiet = bool(status["settled_measurement"])
        if quiet and in_task_band(received):
            self._retry_in_band_home = True
            return None
        if not quiet or received >= TARGET_ML - TOLERANCE_ML:
            self._retry_blocked_reason = "retry20_requires_actual_quiet_undershoot"
            return None
        if self._cycle_count == 0:
            action = self._fixed_cycle_action(status)
            self._retry_selected_actual_approach_deg = FIRST_APPROACH_DEG
            return action
        if self._cycle_count != 1 or not self.profile.enable_retry:
            self._retry_blocked_reason = "retry20_declared_cycle_budget_finished"
            return None
        observed = self._observation(float(t), status)
        u = self.source_time(float(t))
        budget = float(self._planned_budget(u, RETRY_APPROACH_DEG))
        peak = self._cycle_observed_peak_spill
        spill_safe = peak is not None and peak <= 1
        fits = math.isfinite(budget) and budget >= 0. and float(t) + budget <= self.deadline
        allowed = bool(quiet and received < TARGET_ML - TOLERANCE_ML and spill_safe and fits)
        reason = ("actual_quiet_undershoot_retry_allowed" if allowed else
                  "retry20_current_spill_evidence_missing_or_excess" if not spill_safe else
                  "retry20_full_home_motion_budget_exceeds_deadline")
        decision = {"physical_time_s": float(t), "completed_cycles_before_decision": self._cycle_count,
            "actual_received_ml": received, "actual_remaining_ml": float(observed["tube_remaining_ml"]),
            "actual_deficit_ml": TARGET_ML - received, "actual_settled_measurement": quiet,
            "deficit_exceeds_half_ml": received < TARGET_ML - TOLERANCE_ML,
            "observed_all_label_peak_spill_particles": peak, "spill_guard_passed": spill_safe,
            "declared_retry_approach_tilt_deg": RETRY_APPROACH_DEG,
            "conservative_full_home_budget_s": budget, "deadline_physical_seconds": self.deadline,
            "remaining_deadline_physical_seconds": self.deadline - float(t),
            "full_home_budget_fits": fits, "retry_allowed": allowed, "reason": reason,
            "decision_is_intent_not_execution": True,
            "actual_execution_evidence_event_type": "response_cycle_start",
            "budget_is_robot_motion_not_predicted_water_dose": True}
        self._retry_decisions.append(dict(decision))
        self._event("retry20_conditional_second_cycle_decision", float(t), status, retry_decision=dict(decision))
        if not allowed:
            self._retry_blocked_reason = reason
            return None
        action = self._fixed_cycle_action(status)
        action["approach_delta_deg"] = RETRY_APPROACH_DEG - float(status["tilt_deg"])
        self._retry_selected_actual_approach_deg = RETRY_APPROACH_DEG
        return action

    def _real_home(self, t, u, status, reason):
        if bool(status["settled_measurement"]) and in_task_band(float(status["received_ml"])):
            self._retry_in_band_home = True
            reason = "actual_settled_retry20_half_ml_closed_interval"
        elif reason == "model_abstained_or_requested_home" and self._retry_blocked_reason:
            reason = self._retry_blocked_reason
        return super()._real_home(t, u, status, reason)

    def update(self, t, positions, labels, velocities=None):
        status = super().update(t, positions, labels, velocities)
        if self._retry_anchor_requested_at_s is not None and self.prefix_anchor_observation is None:
            self.prefix_anchor_observation = {**self._observation(float(t), status), **self._metadata(),
                "home_requested_physical_time_s": self._retry_anchor_requested_at_s,
                "coarse_cut_tilt_deg": float(self._coarse_cut_angle)}
        if self._cycle_count > self.profile.maximum_cycles or len(self._retry_decisions) > 1:
            raise RuntimeError("Retry20 exceeded its fixed first-cycle/conditional-retry budget")
        status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
            prefix_anchor_observation=self.prefix_anchor_observation,
            retry_decisions=[dict(item) for item in self._retry_decisions],
            retry_actual_dose_in_closed_band=in_task_band(float(status["received_ml"])),
            retry_actual_settled_home_decision=self._retry_in_band_home,
            original_strict_actual_dose_in_closed_band=cycle.actual_dose_in_band(TARGET_ML, status["received_ml"]),
            response_collection_only=False, **self._metadata())
        status["policy_action"] = {**status["policy_action"], **self._metadata(),
            "retry_actual_dose_in_closed_band": status["retry_actual_dose_in_closed_band"],
            "retry_actual_settled_home_decision": self._retry_in_band_home}
        self.status = status
        return dict(status)
