"""CPU prototype: C2 joint-space shortcut with full rigid tube orientation.

This independent route does not replace fast_geometry or run liquid physics.
u is the nominal direct-path clock. A controller must still validate actual
joint acceleration when changing its rate. Full-R containment is mandatory.
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


_REFERENCE_PATH = Path(__file__).resolve().with_name("fast_geometry.py")
REFERENCE_GEOMETRY_SHA256 = "7a259763a3f12a9c0866fed02226a76e777d3ff654846954cd777270875b10ef"
if hashlib.sha256(_REFERENCE_PATH.read_bytes()).hexdigest() != REFERENCE_GEOMETRY_SHA256:
    raise ValueError("The direct-path experiment requires its explicit reference geometry version.")
_spec = importlib.util.spec_from_file_location("direct_private_reference_geometry", _REFERENCE_PATH)
_REFERENCE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _REFERENCE
exec(compile(_REFERENCE_PATH.read_text(encoding="utf-8"), str(_REFERENCE_PATH), "exec"), _REFERENCE.__dict__)
_STATIC = _REFERENCE._OLD
_CACHE = _REFERENCE._CACHE
CACHE_FILE, CACHE_SHA256 = _REFERENCE.CACHE_FILE, _REFERENCE.CACHE_SHA256
JOINT_NAMES = _REFERENCE.JOINT_NAMES
JOINT_VELOCITY_LIMITS_RAD_S = _REFERENCE.JOINT_VELOCITY_LIMITS_RAD_S.copy()
JOINT_LOWER_LIMITS_RAD = _REFERENCE.JOINT_LOWER_LIMITS_RAD.copy()
JOINT_UPPER_LIMITS_RAD = _REFERENCE.JOINT_UPPER_LIMITS_RAD.copy()
RAMP_SECONDS = .20
VELOCITY_FRACTION = .95
JOINT_ACCELERATION_CAP_RAD_S2 = 25.0
Q_HOME = _REFERENCE.q(0.0)
Q_78 = _REFERENCE.q(_REFERENCE.u_for_tilt(78.0))
Q_125 = _REFERENCE.q(_REFERENCE.U_MAX)


def __getattr__(name):
    return getattr(_STATIC, name)


def _duration(a, b):
    delta = np.abs(b - a)
    cruise_time = float(max(delta / (VELOCITY_FRACTION * JOINT_VELOCITY_LIMITS_RAD_S)))
    acceleration_time = float(max(delta)) * math.pi / (2 * RAMP_SECONDS * JOINT_ACCELERATION_CAP_RAD_S2)
    return max(2 * RAMP_SECONDS, RAMP_SECONDS + cruise_time, RAMP_SECONDS + acceleration_time)


PREPARE_END_U = _duration(Q_HOME, Q_78)
DOSE_LEG_SECONDS = _duration(Q_78, Q_125)
U_MAX = DURATION = PREPARE_END_U + DOSE_LEG_SECONDS
UPRIGHT_U = 0.0


def _clip(u):
    u = float(u)
    if not math.isfinite(u):
        raise ValueError("Direct path coordinate must be finite.")
    return float(np.clip(u, 0, U_MAX))


def _progress(t, duration):
    t = float(np.clip(t, 0, duration))
    r, peak = RAMP_SECONDS, 1 / (duration - RAMP_SECONDS)
    if t <= r:
        x = math.pi * t / r
        return (.5 * peak * (t - r * math.sin(x) / math.pi),
                .5 * peak * (1 - math.cos(x)), .5 * peak * math.pi / r * math.sin(x))
    if t < duration - r:
        return peak * (t - r / 2), peak, 0.0
    left = duration - t
    x = math.pi * left / r
    return (1 - .5 * peak * (left - r * math.sin(x) / math.pi),
            .5 * peak * (1 - math.cos(x)), -.5 * peak * math.pi / r * math.sin(x))


def _joint_state(u):
    u = _clip(u)
    if u <= PREPARE_END_U:
        a, b, t, duration = Q_HOME, Q_78, u, PREPARE_END_U
    else:
        a, b, t, duration = Q_78, Q_125, u - PREPARE_END_U, DOSE_LEG_SECONDS
    position, velocity, acceleration = _progress(t, duration)
    delta = b - a
    return a + delta * position, delta * velocity, delta * acceleration


def joint_positions(u):
    return _joint_state(u)[0]


q = joint_positions


def joint_derivative(u):
    return _joint_state(u)[1]


def joint_second_derivative(u):
    return _joint_state(u)[2]


def _hand_matrix(q):
    matrix = _CACHE["base_world_matrix_col"].copy()
    for origin, angle in zip(_CACHE["origin_matrices_col"], q):
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        rotation = np.array(((cosine, -sine, 0, 0), (sine, cosine, 0, 0),
                             (0, 0, 1, 0), (0, 0, 0, 1)), dtype=float)
        matrix = matrix @ origin @ rotation
    return matrix @ _CACHE["hand_fixed_matrix_col"]


_INITIAL_TUBE = np.eye(4)
_INITIAL_TUBE[:3, 3] = _STATIC.tube_pose(0.0)[0]
_HAND_TO_TUBE = np.linalg.inv(_hand_matrix(Q_HOME)) @ _INITIAL_TUBE


def hand_fk(u):
    matrix = _hand_matrix(joint_positions(u))
    return matrix[:3, 3].copy(), matrix[:3, :3].copy()


def tube_matrix(u):
    if _clip(u) == 0.0:
        return _INITIAL_TUBE.copy()
    return _hand_matrix(joint_positions(u)) @ _HAND_TO_TUBE


def tube_rotation(u):
    return tube_matrix(u)[:3, :3].copy()


def _quaternion_xyzw(rotation):
    # Symmetric eigenproblem avoids coordinate-axis singular branches.
    r = rotation
    k = np.array(((r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]),
                  (r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]),
                  (r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]),
                  (r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)))) / 3
    _, vectors = np.linalg.eigh(k)
    result = vectors[:, -1]
    return result if result[3] >= 0 else -result


def tube_pose(u):
    matrix = tube_matrix(u)
    return matrix[:3, 3].copy(), _quaternion_xyzw(matrix[:3, :3])


def hand_pose(u):
    position, rotation = hand_fk(u)
    xyzw = _quaternion_xyzw(rotation)
    return position, xyzw[[3, 0, 1, 2]]


def mouth_pose(u):
    matrix = tube_matrix(u)
    mouth = matrix[:3, 3] + matrix[:3, :3] @ np.array([0., 0., _STATIC.TUBE_LENGTH])
    return mouth, math.acos(float(np.clip(matrix[2, 2], -1, 1)))


def tilt_deg(u):
    return math.degrees(mouth_pose(u)[1])


def u_for_tilt(degrees):
    target = float(degrees)
    if target == 78.0:
        return PREPARE_END_U
    if target == 0.0:
        return 0.0
    if not 0 < target <= 125:
        raise ValueError("Direct path tilt must be in [0,125] degrees.")
    low, high = 0.0, U_MAX
    for _ in range(48):
        middle = .5 * (low + high)
        if tilt_deg(middle) < target:
            low = middle
        else:
            high = middle
    return .5 * (low + high)


def angle_derivative(u):
    u = _clip(u)
    low, high = max(0.0, u - 1e-5), min(U_MAX, u + 1e-5)
    return (mouth_pose(high)[1] - mouth_pose(low)[1]) / (high - low)


def phase(u):
    return "ready" if _clip(u) == 0 else "joint_transfer_to_78" if _clip(u) <= PREPARE_END_U else "direct_dose_tilt"


def initial_water_points(spacing=None):
    return _STATIC.initial_water_points(_STATIC.PARTICLE_SPACING if spacing is None else spacing)


def classify_water(positions, groups, u, spacing=None):
    """Original mass partitions, with full-R tube containment instead of Ry."""
    points, labels = np.asarray(positions, dtype=float), np.asarray(groups, dtype=np.int32)
    spacing = _STATIC.PARTICLE_SPACING if spacing is None else float(spacing)
    if (points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),)
            or not np.isfinite(points).all() or np.any((labels != 0) & (labels != 1))
            or not math.isfinite(spacing) or spacing <= 0):
        raise ValueError("Finite Nx3 points, fixed 0/1 labels and positive spacing are required.")
    delta = points - np.asarray(_STATIC.BEAKER_CENTER)
    beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= _STATIC.BEAKER_RADIUS ** 2)
              & (delta[:, 2] >= 0) & (delta[:, 2] <= _STATIC.BEAKER_HEIGHT))
    transform = tube_matrix(u)
    tube = _STATIC._cavity_mask((points - transform[:3, 3]) @ transform[:3, :3])
    free = ~(beaker | tube)
    airborne = free & (points[:, 2] > _STATIC.TABLE_TOP_Z + spacing)
    spilled = free & ~airborne
    initial_tube = labels == 1
    source_count = int(np.count_nonzero(initial_tube))
    received = int(np.count_nonzero(initial_tube & beaker))
    quantum = spacing ** 3 * 1e6
    return {"time": _clip(u), "phase": phase(u), "total": len(points),
            "beaker": int(np.count_nonzero(beaker)), "tube": int(np.count_nonzero(tube)),
            "airborne": int(np.count_nonzero(airborne)), "spilled": int(np.count_nonzero(spilled)),
            "below_table": int(np.count_nonzero(free & (points[:, 2] < _STATIC.TABLE_TOP_Z - spacing))),
            "initial_tube": source_count, "tube_to_beaker": received,
            "tube_remaining": int(np.count_nonzero(initial_tube & tube)),
            "tube_airborne": int(np.count_nonzero(initial_tube & airborne)),
            "tube_spilled": int(np.count_nonzero(initial_tube & spilled)),
            "tube_transfer_fraction": received / source_count if source_count else 0.0,
            "total_particle_volume_ml": len(points) * quantum, "initial_tube_volume_ml": source_count * quantum,
            "beaker_volume_ml": int(np.count_nonzero(beaker)) * quantum,
            "tube_volume_ml": int(np.count_nonzero(tube)) * quantum}


MOTION_KNOTS = tuple((u, tuple(mouth_pose(u)[0]), tilt_deg(u)) for u in (0., PREPARE_END_U, U_MAX))
