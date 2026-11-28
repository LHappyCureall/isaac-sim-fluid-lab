"""One-factor first-angle40mL controller after frozen24.8deg/s coarse dosing.

Only the FIRST approach angle may vary among87,87.05 and87.1degrees.
COARSE dosing remains24.8degrees/physical-second, retry87.2 and all other coarse/fine parameters,
actual settling feedback and the24s original physical deadline stay fixed.
This new module does not modify its frozen whole-cycle ancestor or v1 source.
It writes no particles, mass, solver, hidden warmup or simulation clock.
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
    raise ValueError("Balanced40 requires the frozen measured whole-cycle controller")
_SPEC = importlib.util.spec_from_file_location("balanced40_frozen_cycle_controller", PARENT_FILE)
cycle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cycle
_SPEC.loader.exec_module(cycle)

POLICY_KIND = "fixed40_balanced_first_angle_conditional_retry_control_v1"
PARAMETER_NAMES = cycle.PARAMETER_NAMES
FIXED_COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 24.8,
                "return_fraction": 1., "tail_lead_s": .26,
                "stop_bias_ml": 10., "max_dose_u_accel": 6.}
FIRST_APPROACH_TILT_DEG = 87.0
COARSE_DOSE_SPEEDS_DEG_S = (24.8,)
FIRST_APPROACH_TILT_DEGREES = (87., 87.05, 87.1)
RETRY_APPROACH_TILT_DEG = 87.2
TARGET_PROFILES = {"40": {"first_approach_tilt_deg": FIRST_APPROACH_TILT_DEG,
                          "retry_approach_tilt_deg": RETRY_APPROACH_TILT_DEG}}
TOLERANCE_ML = .5
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _target(value):
    if type(value) not in (int, float) or value != 40.:
        raise ValueError("This balanced first-angle experiment supports exactly40mL")
    return float(value)


def _coarse(value, selected_speed):
    if (type(selected_speed) not in (int, float) or not math.isfinite(selected_speed)
            or selected_speed not in COARSE_DOSE_SPEEDS_DEG_S):
        raise ValueError("Balanced40 fixes coarse dosing at exactly24.8degrees/physical-second")
    expected = {**FIXED_COARSE, "dose_tilt_deg_s": float(selected_speed)}
    if value is None:
        return expected
    if (not isinstance(value, dict) or set(value) != set(expected)
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   or value[key] != fixed for key, fixed in expected.items())):
        raise ValueError("Balanced40 fixes all six coarse parameters, including24.8degrees/physical-second")
    return dict(expected)


def _angles(target_ml, first, retry):
    _target(target_ml)
    if type(first) not in (int, float) or not math.isfinite(first) or first not in FIRST_APPROACH_TILT_DEGREES:
        raise ValueError("Balanced40 accepts exactly87,87.05 or87.1degrees for the first approach")
    if type(retry) not in (int, float) or not math.isfinite(retry) or retry != RETRY_APPROACH_TILT_DEG:
        raise ValueError("Balanced40 keeps the conditional retry at exactly87.2degrees")
    return float(first), float(retry)


def in_task_band(target_ml, received_ml):
    target = _target(target_ml)
    if type(received_ml) not in (int, float) or not math.isfinite(received_ml):
        raise ValueError("Balanced40 requires finite actual received water")
    return target - TOLERANCE_ML <= received_ml <= target + TOLERANCE_ML


class CorrectionPolicy(cycle.CorrectionPolicy):
    name = POLICY_KIND

    def __init__(self, coarse=None, target_ml=40., enable_retry=True, deadline_s=24.,
                 first_approach_tilt_deg=87.0, retry_approach_tilt_deg=87.2,
                 coarse_dose_tilt_deg_s=24.8, profile_id=None, profile_sha256=None, condition_seed=None):
        self.target_ml = _target(target_ml)
        if enable_retry is not True:
            raise ValueError("Balanced40 keeps conditional retry enabled for both roles")
        if type(deadline_s) not in (int, float) or deadline_s != 24.:
            raise ValueError("Balanced40 keeps the24s original physical clock")
        first, retry = _angles(self.target_ml, first_approach_tilt_deg, retry_approach_tilt_deg)
        if (profile_id is None) != (profile_sha256 is None):
            raise ValueError("External profile identity and raw JSON SHA must be provided together")
        if profile_id is not None:
            if (not isinstance(profile_id, str) or not profile_id or len(profile_id) > 160
                    or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in profile_id)
                    or not isinstance(profile_sha256, str) or len(profile_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in profile_sha256)):
                raise ValueError("A safe profile identity and lowercase SHA256 are required")
        if condition_seed is not None and (type(condition_seed) is not int or not 0 <= condition_seed < 2 ** 31):
            raise ValueError("Condition seed must be the explicit nonnegative integer from the runner")
        self.coarse_dose_tilt_deg_s = float(_coarse(coarse, coarse_dose_tilt_deg_s)["dose_tilt_deg_s"])
        self.condition_seed = condition_seed
        self.enable_retry = enable_retry
        self.first_approach_tilt_deg, self.retry_approach_tilt_deg = first, retry
        self.profile_id, self.profile_sha256 = profile_id, profile_sha256
        super().__init__(_coarse(coarse, self.coarse_dose_tilt_deg_s), deadline_s=24.,
            approach_tilt_deg=first, approach_tilt_deg_s=6., action_delta_deg=.03,
            action_tilt_deg_s=.6, hold_seconds=.1, backoff_delta_deg=.5,
            backoff_tilt_deg_s=4., anchor_return_tilt_deg_s=6., quiet_seconds=.25,
            maximum_measurement_hold_s=1., maximum_cycles=2)
        self.params.update(experiment_kind=POLICY_KIND, enable_retry=enable_retry,
            condition_seed=condition_seed, condition_seed_owned_by_runner=True,
            only_coarse_dose_speed_varies=False, only_first_angle_varies=True,
            only_first_approach_angle_varies=True, coarse_dose_tilt_deg_s=self.coarse_dose_tilt_deg_s,
            fixed_coarse_dose_tilt_deg_s=24.8,
            first_approach_tilt_choices_deg=list(FIRST_APPROACH_TILT_DEGREES),
            fixed_retry_approach_tilt_deg=RETRY_APPROACH_TILT_DEG,
            first_and_retry_angles_fixed=False,
            enable_correction=True, requested_target_ml=self.target_ml,
            first_approach_tilt_deg=first, retry_approach_tilt_deg=retry,
            target_profile_key=str(int(self.target_ml)), target_profile_id=profile_id,
            target_profile_sha256=profile_sha256, external_profile_hash_check_owned_by_runner=True,
            retry_actual_deficit_threshold_ml=TOLERANCE_ML,
            retry_maximum_all_label_peak_spill_particles=1, retry_requires_actual_quiet=True,
            retry_requires_full_home_budget=True, retry_absolute_tolerance_ml=TOLERANCE_ML,
            actual_measurement_target_unchanged=True, coarse_control_target_offset_ml=0.,
            all_motion_speeds_fixed=True, all_other_motion_speeds_fixed=True, all_fine_motion_speeds_fixed=True,
            no_test_based_angle_or_strategy_selection=True, coarse_stop_prediction_rule_unchanged=True,
            settled_actual_volume_decides_home=True, original_strict_band_unchanged=True,
            unchanged_precision_v2_controller=False)


class TargetPourController(cycle.TargetPourController):
    def __init__(self, geometry, target_ml=40., spacing=.0027, policy=None):
        actual = _target(target_ml)
        if type(spacing) not in (int, float) or spacing != .0027:
            raise ValueError("Balanced40 keeps the2.7mm actual particles")
        if not isinstance(policy, CorrectionPolicy) or policy.target_ml != actual:
            raise ValueError("The declared profile must match the actual40mL goal")
        super().__init__(geometry, actual, spacing, policy)

    def reset(self, target_ml=None):
        if target_ml is not None and _target(target_ml) != self.profile.target_ml:
            raise ValueError("Reset may not relabel the profile's actual requested goal")
        super().reset(target_ml)
        self._retry_anchor_requested_at_s = None
        self._retry_in_band_home = False
        self._retry_blocked_reason = None
        self._retry_decisions = []
        self._retry_selected_approach_deg = None
        self.prefix_anchor_observation = None
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params), **self._metadata())
        return dict(self.status)

    def _metadata(self):
        actual_angles = [float(event["approach_tilt_deg"]) for event in self.response_events
                         if event["type"] == "response_cycle_start"]
        return {"actual_target_ml": float(self.target_ml), "control_target_ml": float(self.target_ml),
            "control_target_offset_ml": 0., "prefix_target_offset_ml": 0.,
            "retry_controller_source_sha256": _SOURCE_SHA256,
            "balanced40_controller_source_sha256": _SOURCE_SHA256,
            "balanced40_condition_seed": self.profile.condition_seed,
            "condition_seed_provided_by_runner": self.profile.condition_seed is not None,
            "only_coarse_dose_speed_varies": False, "only_first_angle_varies": True,
            "only_first_approach_angle_varies": True,
            "coarse_dose_tilt_deg_s": self.profile.coarse_dose_tilt_deg_s,
            "fixed_coarse_dose_tilt_deg_s": 24.8,
            "fixed_retry_approach_tilt_deg": RETRY_APPROACH_TILT_DEG,
            "retry_profile_key": str(int(self.target_ml)), "retry_profile_id": self.profile.profile_id,
            "retry_profile_sha256": self.profile.profile_sha256,
            "enable_retry": self.profile.enable_retry, "retry_enabled": self.profile.enable_retry,
            "retry_first_approach_tilt_deg": self.profile.first_approach_tilt_deg,
            "retry_second_approach_tilt_deg": self.profile.retry_approach_tilt_deg,
            "retry_maximum_cycles": self.profile.maximum_cycles,
            "retry_selected_approach_tilt_deg": self._retry_selected_approach_deg,
            "retry_actual_approach_tilt_deg": actual_angles[-1] if actual_angles else None,
            "retry_actual_cycle_approach_tilt_degrees": actual_angles,
            "retry_absolute_tolerance_ml": TOLERANCE_ML, "actual_measurement_target_unchanged": True,
            "coarse_motion_parameters_fixed": True,
            "all_motion_speeds_fixed": True, "all_other_motion_speeds_fixed": True,
            "all_fine_motion_speeds_fixed": True}

    def _start_home(self, t, u):
        if self._retry_anchor_requested_at_s is None:
            self._retry_anchor_requested_at_s = float(t)
        return super()._start_home(t, u)

    def _choose_cycle_action(self, t, status):
        target, received = float(self.target_ml), float(status["received_ml"])
        quiet = bool(status["settled_measurement"])
        if quiet and in_task_band(target, received):
            self._retry_in_band_home = True
            return None
        if not quiet or received >= target - TOLERANCE_ML:
            self._retry_blocked_reason = "balanced40_requires_actual_quiet_undershoot"
            return None
        if self._cycle_count == 0:
            self._retry_selected_approach_deg = self.profile.first_approach_tilt_deg
            return self._fixed_cycle_action(status)
        if self._cycle_count != 1 or not self.profile.enable_retry:
            self._retry_blocked_reason = "balanced40_declared_cycle_budget_finished"
            return None
        observed = self._observation(float(t), status)
        angle = self.profile.retry_approach_tilt_deg
        u = self.source_time(float(t))
        budget = float(self._planned_budget(u, angle))
        peak = self._cycle_observed_peak_spill
        spill_safe = peak is not None and peak <= 1
        fits = math.isfinite(budget) and budget >= 0. and float(t) + budget <= self.deadline
        allowed = bool(quiet and received < target - TOLERANCE_ML and spill_safe and fits)
        reason = ("actual_quiet_target_undershoot_retry_allowed" if allowed else
                  "balanced40_current_spill_evidence_missing_or_excess" if not spill_safe else
                  "balanced40_full_home_motion_budget_exceeds_deadline")
        decision = {"physical_time_s": float(t), "declared_target_ml": target,
            "condition_seed": self.profile.condition_seed,
            "declared_coarse_dose_tilt_deg_s": self.profile.coarse_dose_tilt_deg_s,
            "target_profile_key": str(int(target)), "target_profile_id": self.profile.profile_id,
            "target_profile_sha256": self.profile.profile_sha256,
            "completed_cycles_before_decision": self._cycle_count,
            "actual_received_ml": received, "actual_remaining_ml": float(observed["tube_remaining_ml"]),
            "actual_deficit_ml": target - received, "actual_settled_measurement": quiet,
            "deficit_exceeds_half_ml": received < target - TOLERANCE_ML,
            "observed_all_label_peak_spill_particles": peak, "spill_guard_passed": spill_safe,
            "declared_retry_approach_tilt_deg": angle, "conservative_full_home_budget_s": budget,
            "deadline_physical_seconds": self.deadline, "remaining_deadline_physical_seconds": self.deadline - float(t),
            "full_home_budget_fits": fits, "retry_allowed": allowed, "reason": reason,
            "decision_is_intent_not_execution": True, "actual_execution_evidence_event_type": "response_cycle_start",
            "budget_is_robot_motion_not_predicted_water_dose": True}
        self._retry_decisions.append(dict(decision))
        self._event("balanced40_conditional_second_cycle_decision", float(t), status, retry_decision=dict(decision))
        if not allowed:
            self._retry_blocked_reason = reason
            return None
        action = self._fixed_cycle_action(status)
        action["approach_delta_deg"] = angle - float(status["tilt_deg"])
        self._retry_selected_approach_deg = angle
        return action

    def _real_home(self, t, u, status, reason):
        if bool(status["settled_measurement"]) and in_task_band(self.target_ml, float(status["received_ml"])):
            self._retry_in_band_home = True
            reason = "actual_settled_balanced40_half_ml_closed_interval"
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
            raise RuntimeError("Balanced40 exceeded its declared first-cycle/conditional-retry budget")
        status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
            prefix_anchor_observation=self.prefix_anchor_observation,
            retry_decisions=[dict(item) for item in self._retry_decisions],
            retry_actual_dose_in_closed_band=in_task_band(self.target_ml, float(status["received_ml"])),
            retry_actual_settled_home_decision=self._retry_in_band_home,
            original_strict_actual_dose_in_closed_band=cycle.actual_dose_in_band(self.target_ml, status["received_ml"]),
            response_collection_only=False, **self._metadata())
        status["policy_action"] = {**status["policy_action"], **self._metadata(),
            "retry_actual_dose_in_closed_band": status["retry_actual_dose_in_closed_band"],
            "retry_actual_settled_home_decision": self._retry_in_band_home}
        self.status = status
        return dict(status)
