"""Original liquid geometry with a joint-constrained path clock, not fluid replay.

u has units of normalized robot-speed seconds, not episode physical seconds.
The controller supplies u(physical_time). This module does not advance clocks,
change liquid particles, or claim that the complete action fits five seconds.
"""

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_FILE = PROJECT_ROOT / "_cache/pour_training/fast_v3/fast_path_smooth_cache.npz"
CACHE_SHA256 = os.environ.get("ISAAC_FAST_PATH_CACHE_SHA256", "")
if len(CACHE_SHA256) != 64 or any(c not in "0123456789abcdef" for c in CACHE_SHA256):
    raise RuntimeError("Rebuild and CPU-verify the local robot path cache, then explicitly set ISAAC_FAST_PATH_CACHE_SHA256 to its new SHA256. The archived research cache is not distributed.")
_GEOMETRY_FILE = Path(__file__).resolve().with_name("geometry.py")

if not CACHE_FILE.is_file():
    raise FileNotFoundError("Prepare the explicit robot path cache first: " + str(CACHE_FILE))
if hashlib.sha256(CACHE_FILE.read_bytes()).hexdigest() != CACHE_SHA256:
    raise ValueError("Fast robot path cache differs from its reviewed SHA256.")
with np.load(CACHE_FILE, allow_pickle=False) as _archive:
    _CACHE = {name: _archive[name].copy() for name in _archive.files}
if hashlib.sha256(_GEOMETRY_FILE.read_bytes()).hexdigest() != str(_CACHE["geometry_sha256"]):
    raise ValueError("Original geometry changed after the fast robot path was prepared.")
if float(_CACHE["source_time_scale"]) != 2.0:
    raise ValueError("The reviewed legacy path uses exactly one private time scale of 2.0.")
if int(_CACHE.get("interpolation_version", 0)) != 2:
    raise ValueError("Fast geometry requires the reviewed C2 joint/C1 source clock cache.")
_spec = importlib.util.spec_from_file_location("fast_private_legacy_geometry", _GEOMETRY_FILE)
_OLD = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _OLD
exec(compile(_GEOMETRY_FILE.read_text(encoding="utf-8"), str(_GEOMETRY_FILE), "exec"), _OLD.__dict__)
_OLD.DURATION *= 2.0
_OLD.MOTION_KNOTS = tuple((2.0 * t, point, angle) for t, point, angle in _OLD.MOTION_KNOTS)

_U = _CACHE["u"]
_SOURCE = _CACHE["source"]
_Q = _CACHE["joint_q"]
assert _U.ndim == 1 and len(_U) == len(_SOURCE) == len(_Q) and np.all(np.diff(_U) > 0)
assert np.isfinite(_Q).all() and np.isfinite(_U).all()
U_MAX = DURATION = float(_U[-1])
SOURCE_MAX = float(_SOURCE[-1])
JOINT_NAMES = tuple(str(name) for name in _CACHE["joint_names"])
JOINT_VELOCITY_LIMITS_RAD_S = _CACHE["velocity_limits_rad_s"].copy()
JOINT_LOWER_LIMITS_RAD = _CACHE["lower_limits_rad"].copy()
JOINT_UPPER_LIMITS_RAD = _CACHE["upper_limits_rad"].copy()
PHASE_NAMES = ("lift", "transfer", "pretilt", "pour")


def __getattr__(name):
    # All static dimensions, mesh constructors and masks retain their original
    # implementations. Only APIs taking the path coordinate are overridden.
    return getattr(_OLD, name)


def _clip_u(u):
    value = float(u)
    if not math.isfinite(value):
        raise ValueError("Path coordinate u must be finite.")
    return float(np.clip(value, 0.0, U_MAX))


def _index(values, value):
    return min(max(int(np.searchsorted(values, value, side="right")) - 1, 0), len(values) - 2)


def _source_evaluate(u, derivative=0):
    value = _clip_u(u)
    index = _index(_U, value)
    h = _U[index + 1] - _U[index]
    x = (value - _U[index]) / h
    coefficients = _CACHE["source_coefficients"][index, :, 0]
    result = 0.0
    for power in range(derivative, len(coefficients)):
        factor = math.factorial(power) / math.factorial(power - derivative)
        result += coefficients[power] * factor * x ** (power - derivative) / h ** derivative
    return float(result)


def source_from_u(u):
    return float(np.clip(_source_evaluate(u), 0.0, SOURCE_MAX))


def u_from_source(source):
    source = float(source)
    if not math.isfinite(source):
        raise ValueError("Legacy source coordinate must be finite.")
    source = float(np.clip(source, 0.0, SOURCE_MAX))
    if source == 0.0:
        return 0.0
    if source == SOURCE_MAX:
        return U_MAX
    index = _index(_SOURCE, source)
    low, high = float(_U[index]), float(_U[index + 1])
    for _ in range(32):
        middle = .5 * (low + high)
        if source_from_u(middle) < source:
            low = middle
        else:
            high = middle
    return .5 * (low + high)


MOTION_KNOTS = tuple((u_from_source(t), point, angle) for t, point, angle in _OLD.MOTION_KNOTS[:5])
UPRIGHT_U = u_from_source(_OLD.MOTION_KNOTS[2][0])


def u_rate_to_source_rate(u, u_rate):
    return float(u_rate) * _source_evaluate(u, 1)


def source_rate_to_u_rate(source, source_rate):
    return float(source_rate) / _source_evaluate(u_from_source(source), 1)


def mouth_pose(u):
    return _OLD.mouth_pose(source_from_u(u))


def tube_pose(u):
    return _OLD.tube_pose(source_from_u(u))


def hand_pose(u):
    return _OLD.hand_pose(source_from_u(u))


def tube_rotation(u):
    return _OLD.rotation_y(mouth_pose(u)[1])


def phase(u):
    return _OLD.phase(source_from_u(u))


def tilt_deg(u):
    return math.degrees(float(mouth_pose(u)[1]))


def u_for_tilt(degrees):
    degrees = float(degrees)
    if not math.isfinite(degrees) or not 0.0 <= degrees <= 125.0:
        raise ValueError("The outbound bank tilt is in [0, 125] degrees.")
    if degrees == 0.0:
        return UPRIGHT_U
    low, high = UPRIGHT_U, U_MAX
    for _ in range(52):
        middle = 0.5 * (low + high)
        if tilt_deg(middle) < degrees:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def angle_derivative(u):
    """d(tilt radians)/du; multiply by signed u_rate for physical rad/s."""
    source = source_from_u(u)
    for start, end in zip(_OLD.MOTION_KNOTS[:4], _OLD.MOTION_KNOTS[1:5]):
        if source <= end[0]:
            x = float(np.clip((source - start[0]) / (end[0] - start[0]), 0.0, 1.0))
            derivative = math.radians(end[2] - start[2]) * 30.0 * x * x * (1.0 - x) ** 2 / (end[0] - start[0])
            return derivative * u_rate_to_source_rate(u, 1.0)
    return 0.0


def _joint_evaluate(source, derivative=0):
    grid = _CACHE["joint_spline_source"]
    index = _index(grid, source)
    h = grid[index + 1] - grid[index]
    x = (source - grid[index]) / h
    coefficients = _CACHE["joint_coefficients"][index]
    result = np.zeros(7)
    for power in range(derivative, len(coefficients)):
        factor = math.factorial(power) / math.factorial(power - derivative)
        result += coefficients[power] * factor * x ** (power - derivative) / h ** derivative
    return result


def joint_positions(u):
    return _joint_evaluate(source_from_u(u))


q = joint_positions


def joint_derivative(u):
    """Continuous dq/du from C2 q(source) and C1 inverse path clock."""
    return _joint_evaluate(source_from_u(u), 1) * _source_evaluate(u, 1)


def joint_second_derivative(u):
    source = source_from_u(u)
    first = _source_evaluate(u, 1)
    return (_joint_evaluate(source, 2) * first * first
            + _joint_evaluate(source, 1) * _source_evaluate(u, 2))


def joint_u_acceleration_bounds(u, u_rate, joint_acceleration_cap=25.0):
    """Signed u-acceleration interval implied by physical joint rad/s^2 caps."""
    first, second = joint_derivative(u), joint_second_derivative(u)
    low, high = -math.inf, math.inf
    for slope, curvature in zip(first, second):
        offset = curvature * float(u_rate) ** 2
        if abs(slope) < 1e-10:
            if abs(offset) > joint_acceleration_cap:
                return 1.0, -1.0
            continue
        values = ((-joint_acceleration_cap - offset) / slope,
                  (joint_acceleration_cap - offset) / slope)
        low, high = max(low, min(values)), min(high, max(values))
    return float(low), float(high)


def hand_fk(u):
    """CPU URDF FK of the cached joint path, for independent grasp auditing."""
    matrix = _CACHE["base_world_matrix_col"].copy()
    for origin, angle in zip(_CACHE["origin_matrices_col"], joint_positions(u)):
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        rotation = np.array(((cosine, -sine, 0, 0), (sine, cosine, 0, 0),
                             (0, 0, 1, 0), (0, 0, 0, 1)), dtype=float)
        matrix = matrix @ origin @ rotation
    matrix = matrix @ _CACHE["hand_fixed_matrix_col"]
    return matrix[:3, 3].copy(), matrix[:3, :3].copy()


def initial_water_points(spacing=None):
    return _OLD.initial_water_points(_OLD.PARTICLE_SPACING if spacing is None else spacing)


def classify_water(positions, groups, u, spacing=None):
    return _OLD.classify_water(positions, groups, source_from_u(u),
                               _OLD.PARTICLE_SPACING if spacing is None else spacing)
