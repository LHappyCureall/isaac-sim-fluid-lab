"""One measured correction cycle for a separate20mL +/-0.5mL curriculum.

The fixed-mouth trajectory, live liquid, coarse profile and all speeds are
unchanged. Only the declared absolute approach angle may be calibrated. The
original strict assessment is retained separately from this new task band.
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
    raise ValueError("Refine20 requires the frozen measured whole-cycle controller")
_SPEC = importlib.util.spec_from_file_location("refine20_frozen_cycle_controller", PARENT_FILE)
cycle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cycle
_SPEC.loader.exec_module(cycle)

POLICY_KIND = "fixed20_settled_single_cycle_angle_calibration_v1"
PARAMETER_NAMES = cycle.PARAMETER_NAMES
APPROACH_ANGLES_DEG = (84.0, 84.1, 84.2)
FIXED_COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
                "return_fraction": 1., "tail_lead_s": .26,
                "stop_bias_ml": 10., "max_dose_u_accel": 6.}
TARGET_ML = 20.
TOLERANCE_ML = .5
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _coarse(value):
    if value is None:
        return dict(FIXED_COARSE)
    if (not isinstance(value, dict) or set(value) != set(FIXED_COARSE)
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   or value[key] != fixed for key, fixed in FIXED_COARSE.items())):
        raise ValueError("Refine20 fixes all six original coarse parameters")
    return dict(FIXED_COARSE)


def in_refine_band(received_ml):
    if type(received_ml) not in (int, float) or not math.isfinite(received_ml):
        raise ValueError("Refine20 requires a finite actual received volume")
    return TARGET_ML - TOLERANCE_ML <= received_ml <= TARGET_ML + TOLERANCE_ML


class CorrectionPolicy(cycle.CorrectionPolicy):
    name = POLICY_KIND

    def __init__(self, coarse=None, approach_tilt_deg=84.1,
                 enable_correction=True, deadline_s=24.):
        if (type(approach_tilt_deg) not in (int, float)
                or not math.isfinite(approach_tilt_deg)
                or approach_tilt_deg not in APPROACH_ANGLES_DEG):
            raise ValueError("Only the three predeclared absolute approach angles are allowed")
        if type(enable_correction) is not bool:
            raise ValueError("enable_correction must be an explicit boolean")
        if type(deadline_s) not in (int, float) or deadline_s != 24.:
            raise ValueError("Refine20 keeps the24s raw physical deadline")
        self.enable_correction = enable_correction
        super().__init__(_coarse(coarse), deadline_s=24.,
            approach_tilt_deg=float(approach_tilt_deg), approach_tilt_deg_s=6.,
            action_delta_deg=.03, action_tilt_deg_s=.6, hold_seconds=.1,
            backoff_delta_deg=.5, backoff_tilt_deg_s=4.,
            anchor_return_tilt_deg_s=6., quiet_seconds=.25,
            maximum_measurement_hold_s=1., maximum_cycles=1)
        self.params.update(experiment_kind=POLICY_KIND, enable_correction=enable_correction,
            refine_task_target_ml=TARGET_ML, refine_absolute_tolerance_ml=TOLERANCE_ML,
            actual_measurement_target_unchanged=True, coarse_control_target_offset_ml=0.,
            all_motion_speeds_fixed=True, only_approach_angle_is_calibrated=True,
            settled_actual_volume_decides_home=True, original_strict_band_unchanged=True,
            response_collection_only=False, unchanged_precision_v2_controller=False)


class TargetPourController(cycle.TargetPourController):
    def __init__(self, geometry, target_ml=20., spacing=.0027, policy=None):
        if type(target_ml) not in (int, float) or target_ml != TARGET_ML:
            raise ValueError("This curriculum has exactly one actual target:20mL")
        if type(spacing) not in (int, float) or spacing != .0027:
            raise ValueError("Refine20 keeps the2.7mm particles and their actual masses")
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Refine20 requires its declared CorrectionPolicy")
        super().__init__(geometry, target_ml, spacing, policy)

    def reset(self, target_ml=None):
        if target_ml is not None and (type(target_ml) not in (int, float) or target_ml != TARGET_ML):
            raise ValueError("Reset may not change the actual20mL goal")
        result = super().reset(target_ml)
        self._refine_anchor_requested_at_s = None
        self._refine_in_band_home = False
        self.prefix_anchor_observation = None
        self.status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
                           **self._metadata())
        return dict(self.status)

    def _metadata(self):
        return {"actual_target_ml": TARGET_ML, "control_target_ml": TARGET_ML,
            "control_target_offset_ml": 0., "prefix_target_offset_ml": 0.,
            "refine_controller_source_sha256": _SOURCE_SHA256,
            "refine_enable_correction": self.profile.enable_correction,
            "enable_correction": self.profile.enable_correction,
            "refine_configured_approach_tilt_deg": self.profile.approach_tilt_deg,
            "refine_maximum_cycles": 1,
            "refine_absolute_tolerance_ml": TOLERANCE_ML,
            "actual_measurement_target_unchanged": True,
            "coarse_motion_parameters_fixed": True}

    def _start_home(self, t, u):
        if self._refine_anchor_requested_at_s is None:
            self._refine_anchor_requested_at_s = float(t)
        if not self.profile.enable_correction:
            # Identical physical coarse reference: bypass both response
            # interceptors and execute the original reverse-prepare home.
            return cycle.response.parent.TargetPourController._start_home(self, t, u)
        return super()._start_home(t, u)

    def _choose_cycle_action(self, t, status):
        if bool(status["settled_measurement"]) and in_refine_band(status["received_ml"]):
            self._refine_in_band_home = True
            return None
        return super()._choose_cycle_action(t, status)

    def _real_home(self, t, u, status, reason):
        if bool(status["settled_measurement"]) and in_refine_band(status["received_ml"]):
            self._refine_in_band_home = True
            reason = "actual_settled_refine_half_ml_closed_interval"
        # The inherited event still records its original strict-band boolean.
        # A new explicit reason separates this curriculum from that criterion.
        return super()._real_home(t, u, status, reason)

    def update(self, t, positions, labels, velocities=None):
        status = super().update(t, positions, labels, velocities)
        if self._refine_anchor_requested_at_s is not None and self.prefix_anchor_observation is None:
            self.prefix_anchor_observation = {**self._observation(float(t), status), **self._metadata(),
                "home_requested_physical_time_s": self._refine_anchor_requested_at_s,
                "coarse_cut_tilt_deg": float(self._coarse_cut_angle)}
        if not self.profile.enable_correction and self.response_events:
            raise RuntimeError("Coarse baseline unexpectedly executed a correction event")
        if self._cycle_count > 1:
            raise RuntimeError("Refine20 exceeded its one actual measured correction cycle")
        status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
            prefix_anchor_observation=self.prefix_anchor_observation,
            refine_actual_dose_in_closed_band=in_refine_band(float(status["received_ml"])),
            refine_actual_settled_home_decision=self._refine_in_band_home,
            original_strict_actual_dose_in_closed_band=cycle.actual_dose_in_band(TARGET_ML, status["received_ml"]),
            response_collection_only=False, **self._metadata())
        status["policy_action"] = {**status["policy_action"], **self._metadata(),
            "refine_actual_dose_in_closed_band": status["refine_actual_dose_in_closed_band"],
            "refine_actual_settled_home_decision": self._refine_in_band_home}
        self.status = status
        return dict(status)
