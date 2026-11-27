"""Independent upright-lift then fixed-mouth geometry, CPU prototype only.

u is seconds of an embedded C2 nominal joint clock: upright 25 mm lift,
joint transport to 78 degrees, then fixed-mouth pure-Ry dosing to 125.
All FK and tube poses use the actual rigid grasp. This module never runs
liquid, edits a particle or claims a five-second feedback result.
"""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "projects/pour_policy_training_v3"
BASE_FILE = Path(__file__).resolve().with_name("mouth_pivot_fast_geometry.py")
BASE_GEOMETRY_SHA256 = "44b11468f1ecfbad56127a1747d7c54848b0b56475c0c3ea0c27b6d20eaaaffb"
DEFAULT_CACHE_FILE = HERE / "mouth_pivot_lift025_z022_cache.npz"
DEFAULT_CACHE_SHA256 = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
BASE_CACHE_FILE = HERE / "mouth_pivot_z022_cache.npz"
BASE_CACHE_SHA256 = "f0dec80ef890b2ef8946627af98be8d736e2a3d684e7f8f7bf2adb120aa46d40"
BASE_IMPORT_CACHE_FILE = HERE / "mouth_pivot_z026_cache.npz"
BASE_IMPORT_CACHE_SHA256 = "86ebd4f519f9341fc39c21b02fea47fb9cebe6117d367da68ef6729391dc1bfd"
_INPUTS = {BASE_FILE: BASE_GEOMETRY_SHA256, BASE_CACHE_FILE: BASE_CACHE_SHA256,
           BASE_IMPORT_CACHE_FILE: BASE_IMPORT_CACHE_SHA256}
for _path, _sha in _INPUTS.items():
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _sha:
        raise ValueError("Upright-lift prototype input SHA256 differs: " + str(_path))
_spec = importlib.util.spec_from_file_location("upright_lift_pinned_pivot_base", BASE_FILE)
_BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_BASE)


class LiftPivotGeometry(_BASE.PivotGeometry):
    def __init__(self, cache_file, expected_sha256):
        self._lift_ready = False
        super().__init__(cache_file, expected_sha256)
        data = self._CACHE
        if (str(data.get("kind")) != "optional_upright_lift_fixed_mouth_not_active_runtime"
                or str(data.get("base_cache_sha256")) != BASE_CACHE_SHA256):
            raise ValueError("The upright-lift runtime requires its own explicitly pinned cache.")
        self._lift_grid = data["lift_progress_grid"].copy()
        self._lift_coeff = data["lift_joint_coefficients"].copy()
        if (self._lift_grid.shape != (41,) or self._lift_coeff.shape != (40, 6, 7)
                or self._lift_grid[0] != 0 or self._lift_grid[-1] != 1
                or np.any(np.diff(self._lift_grid) <= 0)):
            raise ValueError("Unexpected upright-lift interpolation dimensions or domain.")
        self.Q_LIFT = data["q_lift_end"].copy()
        self.LIFT_HEIGHT_M = float(data["lift_height_m"])
        self.LIFT_END_U = float(data["lift_duration_s"])
        self.TRANSFER_SECONDS = float(data["lifted_to_78_duration_s"])
        if (not .02 <= self.LIFT_HEIGHT_M <= .03 or self.LIFT_END_U < 2 * self.RAMP_SECONDS
                or self.TRANSFER_SECONDS < 2 * self.RAMP_SECONDS):
            raise ValueError("Upright-lift timing/height lies outside the CPU preparation protocol.")
        self.PREPARE_END_U = self.LIFT_END_U + self.TRANSFER_SECONDS
        self.U_MAX = self.DURATION = self.PREPARE_END_U + self.DOSE_LEG_SECONDS
        self.__file__ = __file__
        self.INPUT_SHA256 = {str(path.relative_to(ROOT)).replace("\\", "/"): sha
                             for path, sha in _INPUTS.items()}
        self.INPUT_SHA256["assets/robot_testtube_pour/scripts/geometry.py"] = _BASE.ORIGINAL_GEOMETRY_SHA256
        self._lift_ready = True
        self.MOTION_KNOTS = tuple((u, tuple(self.mouth_pose(u)[0]), self.tilt_deg(u))
                                  for u in (0., self.LIFT_END_U, self.PREPARE_END_U, self.U_MAX))

    def _lift_q(self, progress, derivative):
        grid = self._lift_grid
        index = int(np.clip(np.searchsorted(grid, progress, side="right") - 1, 0, len(grid) - 2))
        h = float(grid[index + 1] - grid[index])
        x = (progress - grid[index]) / h
        value = np.zeros(7)
        for power in range(derivative, 6):
            factor = math.factorial(power) / math.factorial(power - derivative)
            value += self._lift_coeff[index, power] * factor * x ** (power - derivative) / h ** derivative
        return value

    def _joint_state(self, u):
        if not self._lift_ready:
            return super()._joint_state(u)
        u = self._clip(u)
        if u <= self.LIFT_END_U:
            p, v, a = _BASE._progress(u, self.LIFT_END_U, self.RAMP_SECONDS)
            first = self._lift_q(p, 1)
            return self._lift_q(p, 0), first * v, self._lift_q(p, 2) * v * v + first * a
        if u <= self.PREPARE_END_U:
            p, v, a = _BASE._progress(u - self.LIFT_END_U, self.TRANSFER_SECONDS, self.RAMP_SECONDS)
            delta = self.Q_78 - self.Q_LIFT
            return self.Q_LIFT + delta * p, delta * v, delta * a
        return super()._joint_state(u)

    def phase(self, u):
        if not self._lift_ready:
            return super().phase(u)
        u = self._clip(u)
        return ("ready" if u == 0 else "upright_25mm_lift" if u <= self.LIFT_END_U else
                "lifted_joint_transfer_to_fixed_mouth_78" if u <= self.PREPARE_END_U else
                "fixed_mouth_dose_tilt")


def load_geometry(cache_file, expected_sha256):
    return LiftPivotGeometry(cache_file, expected_sha256)


_DEFAULT = None


def __getattr__(name):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_geometry(DEFAULT_CACHE_FILE, DEFAULT_CACHE_SHA256)
    return getattr(_DEFAULT, name)
