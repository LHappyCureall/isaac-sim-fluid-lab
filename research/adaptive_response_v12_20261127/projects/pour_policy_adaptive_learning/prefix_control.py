"""Bounded coarse-prefix system identification, without correction cycles.

Only the internal coarse cut target receives a declared mL offset.  The actual
requested target, mass/volume truth, particle arrays, motion speeds and 24 s
clock remain unchanged. At the first actually quiet 78-degree anchor the robot
returns directly to original home and completes the inherited quiet criterion.
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PARENT_FILE = HERE / "cycle_control.py"
PARENT_SHA256 = "927d2b7600ec1d0d40a34296659c29bb4727e73969a1d9810648347972a79daa"
if hashlib.sha256(PARENT_FILE.read_bytes()).hexdigest() != PARENT_SHA256:
    raise ValueError("Prefix calibration requires the frozen whole-cycle motion controller")
_SPEC = importlib.util.spec_from_file_location("prefix_frozen_cycle_controller", PARENT_FILE)
cycle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cycle
_SPEC.loader.exec_module(cycle)

POLICY_KIND = "coarse_prefix_control_target_offset_calibration_v1"
PARAMETER_NAMES = cycle.PARAMETER_NAMES
FIXED_COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
                "return_fraction": 1., "tail_lead_s": .26,
                "stop_bias_ml": 10., "max_dose_u_accel": 6.}
DirectPolicy = cycle.response.parent.base.DirectPolicy
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _offset(value):
    if (type(value) not in (int, float) or not math.isfinite(value) or not -2. <= value <= 2.):
        raise ValueError("Internal control target offset must be finite within[-2,+2]mL")
    return float(value)


def _fixed_coarse(coarse):
    if (not isinstance(coarse, dict) or set(coarse) != set(FIXED_COARSE)
            or any(type(coarse[key]) not in (int, float) or not math.isfinite(coarse[key])
                   or float(coarse[key]) != fixed for key, fixed in FIXED_COARSE.items())):
        raise ValueError("Prefix calibration fixes all six original coarse motion/control parameters")
    return dict(FIXED_COARSE)


class PrefixCoarsePolicy(DirectPolicy):
    """Read-only observation adapter; the caller's actual target is untouched."""

    def __init__(self, coarse, control_target_offset_ml):
        super().__init__(_fixed_coarse(coarse))
        self.control_target_offset_ml = _offset(control_target_offset_ml)

    def act(self, observation):
        copied = dict(observation)
        actual = copied["target_ml"]
        if type(actual) not in (int, float) or not math.isfinite(actual) or not 20. <= actual <= 40.:
            raise ValueError("Prefix actual measurement goal must remain within20--40mL")
        control_target = float(actual) + self.control_target_offset_ml
        copied["target_ml"] = control_target
        action = super().act(copied)
        return {**action, "actual_target_ml": float(actual), "control_target_ml": control_target,
                "prefix_target_offset_ml": self.control_target_offset_ml,
                "control_target_offset_ml": self.control_target_offset_ml,
                "internal_target_offset_only_not_measurement_calibration": True}


class CorrectionPolicy(cycle.CorrectionPolicy):
    name = POLICY_KIND

    def __init__(self, coarse, control_target_offset_ml=0., deadline_s=24.):
        if type(deadline_s) not in (int, float) or deadline_s != 24.:
            raise ValueError("Prefix calibration cannot alter the24s actual action deadline")
        fixed = _fixed_coarse(coarse)
        self.control_target_offset_ml = _offset(control_target_offset_ml)
        super().__init__(fixed, deadline_s=24., maximum_cycles=1)
        self.params = {**self.params, "experiment_kind": POLICY_KIND,
                       "control_target_offset_ml": self.control_target_offset_ml,
                       "prefix_target_offset_ml": self.control_target_offset_ml,
                       "prefix_only_no_approach_or_micro_cycles": True,
                       "actual_measurement_target_unchanged": True,
                       "coarse_motion_parameters_fixed": True}


class TargetPourController(cycle.TargetPourController):
    def __init__(self, geometry, target_ml=30., spacing=.0027, policy=None):
        if not isinstance(policy, CorrectionPolicy):
            raise ValueError("Prefix calibration requires its own CorrectionPolicy")
        super().__init__(geometry, target_ml, spacing, policy)

    def reset(self, target_ml=None):
        result = super().reset(target_ml)
        self.policy = PrefixCoarsePolicy(self.profile.coarse, self.profile.control_target_offset_ml)
        self._prefix_home_requested_at_s = None
        self.prefix_anchor_observation = None
        self.status.update(policy_name=POLICY_KIND, **self._target_metadata())
        return dict(self.status)

    def _target_metadata(self):
        return {"actual_target_ml": float(self.target_ml),
                "control_target_ml": float(self.target_ml) + self.profile.control_target_offset_ml,
                "prefix_target_offset_ml": self.profile.control_target_offset_ml,
                "control_target_offset_ml": self.profile.control_target_offset_ml,
                "prefix_controller_source_sha256": _SOURCE_SHA256,
                "actual_measurement_target_unchanged": True}

    def _start_home(self, t, u):
        # Bypass both interceptors. No approach/micro/re-correction is allowed.
        self._prefix_home_requested_at_s = float(t)
        return cycle.response.parent.TargetPourController._start_home(self, t, u)

    def update(self, t, positions, labels, velocities=None):
        status = super().update(t, positions, labels, velocities)
        if (self._prefix_home_requested_at_s is not None and self.prefix_anchor_observation is None):
            self.prefix_anchor_observation = {**self._observation(float(t), status),
                                              **self._target_metadata(),
                                              "home_requested_physical_time_s": self._prefix_home_requested_at_s,
                                              "coarse_cut_tilt_deg": float(self._coarse_cut_angle)}
        if self.response_events:
            raise RuntimeError("Prefix-only controller unexpectedly entered a response action")
        status.update(policy_name=POLICY_KIND, policy_params=dict(self.profile.params),
                      prefix_only_calibration=True, response_collection_only=False,
                      response_model_used=False, response_events=[], response_cycle_count=0,
                      prefix_anchor_observation=self.prefix_anchor_observation,
                      **self._target_metadata())
        status["policy_action"] = {**status["policy_action"], **self._target_metadata(),
                                   "prefix_only_calibration": True}
        self.status = status
        return dict(status)
