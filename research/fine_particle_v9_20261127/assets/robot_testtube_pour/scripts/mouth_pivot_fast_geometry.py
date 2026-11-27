"""Independent CPU-prepared fixed-mouth route; no liquid result is implied.

The nominal coordinate u is seconds of the C2 joint clock. The mouth is fixed
during the 78--125 degree dose leg, to offline IK/interpolation precision.
Changing u rate still requires checking actual qdot and qddot. Initial water,
cup and tube shape are the original experiment's, without preconditioning.
"""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_FILE = ROOT / "projects/pour_policy_training_v3/mouth_pivot_z026_cache.npz"
DEFAULT_CACHE_SHA256 = "86ebd4f519f9341fc39c21b02fea47fb9cebe6117d367da68ef6729391dc1bfd"
ORIGINAL_GEOMETRY_SHA256 = "a0071f92d282c0bb10d8c2e121fd65e5a91c08bf40b9188b3a4594fc9bb5cfda"
_STATIC_FILE = Path(__file__).resolve().with_name("geometry.py")
if hashlib.sha256(_STATIC_FILE.read_bytes()).hexdigest() != ORIGINAL_GEOMETRY_SHA256:
    raise ValueError("The pivot experiment requires its explicit original geometry version.")
_spec = importlib.util.spec_from_file_location("mouth_pivot_private_original", _STATIC_FILE)
_STATIC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_STATIC)


def _progress(t, duration, ramp):
    t = float(np.clip(t, 0, duration))
    peak = 1 / (duration - ramp)
    if t <= ramp:
        x = math.pi * t / ramp
        return (.5 * peak * (t - ramp * math.sin(x) / math.pi),
                .5 * peak * (1 - math.cos(x)), .5 * peak * math.pi / ramp * math.sin(x))
    if t < duration - ramp:
        return peak * (t - ramp / 2), peak, 0.0
    left = duration - t
    x = math.pi * left / ramp
    return (1 - .5 * peak * (left - ramp * math.sin(x) / math.pi),
            .5 * peak * (1 - math.cos(x)), -.5 * peak * math.pi / ramp * math.sin(x))


def _quaternion_xyzw(r):
    matrix = np.array(((r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]),
                       (r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]),
                       (r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]),
                       (r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)))) / 3
    _, vectors = np.linalg.eigh(matrix)
    result = vectors[:, -1]
    return result if result[3] >= 0 else -result


class PivotGeometry:
    """Load a separately checked height, requiring its explicit raw SHA256."""

    def __init__(self, cache_file, expected_sha256):
        self.CACHE_FILE = Path(cache_file).resolve()
        self.CACHE_SHA256 = hashlib.sha256(self.CACHE_FILE.read_bytes()).hexdigest()
        if self.CACHE_SHA256 != expected_sha256:
            raise ValueError("Fixed-mouth path cache SHA256 does not match.")
        with np.load(self.CACHE_FILE, allow_pickle=False) as archive:
            self._CACHE = {key: archive[key].copy() for key in archive.files}
        data = self._CACHE
        if str(data["original_geometry_sha256"]) != ORIGINAL_GEOMETRY_SHA256:
            raise ValueError("Fixed-mouth cache and initial geometry disagree.")
        for name, value in data.items():
            if value.dtype.kind in "fci" and not np.isfinite(value).all():
                raise ValueError("Fixed-mouth cache contains nonfinite values: " + name)
        self._STATIC = _STATIC
        self.__file__ = __file__
        self.JOINT_NAMES = tuple(str(x) for x in data["joint_names"])
        self.JOINT_VELOCITY_LIMITS_RAD_S = data["velocity_limits_rad_s"].copy()
        self.JOINT_LOWER_LIMITS_RAD = data["lower_limits_rad"].copy()
        self.JOINT_UPPER_LIMITS_RAD = data["upper_limits_rad"].copy()
        self.Q_HOME = data["q_home"].copy()
        self.Q_78, self.Q_125 = data["theta_joint_q"][[0, -1]].copy()
        self._HAND_TO_TUBE = data["hand_to_tube"].copy()
        self.PIVOT_M = data["pivot_m"].copy()
        self.RAMP_SECONDS = float(data["ramp_s"])
        self.PREPARE_END_U = float(data["prepare_duration_s"])
        self.DOSE_LEG_SECONDS = float(data["dose_duration_s"])
        self.U_MAX = self.DURATION = self.PREPARE_END_U + self.DOSE_LEG_SECONDS
        self.JOINT_ACCELERATION_CAP_RAD_S2 = 25.0
        self.UPRIGHT_U = 0.0
        self._INITIAL_TUBE = np.eye(4)
        self._INITIAL_TUBE[:3, 3] = _STATIC.tube_pose(0.0)[0]
        self.MOTION_KNOTS = tuple((u, tuple(self.mouth_pose(u)[0]), self.tilt_deg(u))
                                  for u in (0., self.PREPARE_END_U, self.U_MAX))

    def __getattr__(self, name):
        return getattr(_STATIC, name)

    def _clip(self, u):
        if not math.isfinite(float(u)):
            raise ValueError("Fixed-mouth path coordinate must be finite.")
        return float(np.clip(u, 0, self.U_MAX))

    def _theta_state(self, u):
        angles = self._CACHE["theta_rad"]
        p, v, a = _progress(self._clip(u) - self.PREPARE_END_U,
                            self.DOSE_LEG_SECONDS, self.RAMP_SECONDS)
        delta = float(angles[-1] - angles[0])
        return float(angles[0]) + delta * p, delta * v, delta * a

    def _qtheta(self, theta, derivative):
        grid, coeff = self._CACHE["theta_rad"], self._CACHE["theta_joint_coefficients"]
        index = int(np.clip(np.searchsorted(grid, theta, side="right") - 1, 0, len(grid) - 2))
        h = float(grid[index + 1] - grid[index])
        x = (theta - grid[index]) / h
        result = np.zeros(7)
        for power in range(derivative, 6):
            factor = math.factorial(power) / math.factorial(power - derivative)
            result += coeff[index, power] * factor * x ** (power - derivative) / h ** derivative
        return result

    def _joint_state(self, u):
        u = self._clip(u)
        if u <= self.PREPARE_END_U:
            p, v, a = _progress(u, self.PREPARE_END_U, self.RAMP_SECONDS)
            delta = self.Q_78 - self.Q_HOME
            return self.Q_HOME + p * delta, v * delta, a * delta
        theta, theta_v, theta_a = self._theta_state(u)
        first = self._qtheta(theta, 1)
        return (self._qtheta(theta, 0), first * theta_v,
                self._qtheta(theta, 2) * theta_v ** 2 + first * theta_a)

    def joint_positions(self, u):
        return self._joint_state(u)[0]

    q = joint_positions

    def joint_derivative(self, u):
        return self._joint_state(u)[1]

    def joint_second_derivative(self, u):
        return self._joint_state(u)[2]

    def _hand_matrix(self, q):
        matrix = self._CACHE["base_world_matrix_col"].copy()
        for origin, angle in zip(self._CACHE["origin_matrices_col"], q):
            c, s = math.cos(float(angle)), math.sin(float(angle))
            matrix = matrix @ origin @ np.array(((c, -s, 0, 0), (s, c, 0, 0),
                                                 (0, 0, 1, 0), (0, 0, 0, 1)), dtype=float)
        return matrix @ self._CACHE["hand_fixed_matrix_col"]

    def hand_fk(self, u):
        matrix = self._hand_matrix(self.joint_positions(u))
        return matrix[:3, 3].copy(), matrix[:3, :3].copy()

    def tube_matrix(self, u):
        return (self._INITIAL_TUBE.copy() if self._clip(u) == 0 else
                self._hand_matrix(self.joint_positions(u)) @ self._HAND_TO_TUBE)

    def tube_rotation(self, u):
        return self.tube_matrix(u)[:3, :3].copy()

    def tube_pose(self, u):
        matrix = self.tube_matrix(u)
        return matrix[:3, 3].copy(), _quaternion_xyzw(matrix[:3, :3])

    def hand_pose(self, u):
        position, rotation = self.hand_fk(u)
        return position, _quaternion_xyzw(rotation)[[3, 0, 1, 2]]

    def mouth_pose(self, u):
        matrix = self.tube_matrix(u)
        mouth = matrix[:3, 3] + matrix[:3, :3] @ np.array((0., 0., _STATIC.TUBE_LENGTH))
        return mouth, math.acos(float(np.clip(matrix[2, 2], -1, 1)))

    def tilt_deg(self, u):
        return math.degrees(self.mouth_pose(u)[1])

    def u_for_tilt(self, degrees):
        target = float(degrees)
        if target == 78.0:
            return self.PREPARE_END_U
        if target == 0.0:
            return 0.0
        if not 0 < target <= 125:
            raise ValueError("Fixed-mouth tilt must be in [0,125] degrees.")
        low, high = 0.0, self.U_MAX
        for _ in range(48):
            middle = .5 * (low + high)
            if self.tilt_deg(middle) < target:
                low = middle
            else:
                high = middle
        return .5 * (low + high)

    def angle_derivative(self, u):
        u = self._clip(u)
        if u > self.PREPARE_END_U:
            return self._theta_state(u)[1]
        low, high = max(0., u - 1e-5), min(self.PREPARE_END_U, u + 1e-5)
        return (self.mouth_pose(high)[1] - self.mouth_pose(low)[1]) / (high - low)

    def phase(self, u):
        return ("ready" if self._clip(u) == 0 else "joint_transfer_to_fixed_mouth_78"
                if self._clip(u) <= self.PREPARE_END_U else "fixed_mouth_dose_tilt")

    def initial_water_points(self, spacing=None):
        return _STATIC.initial_water_points(_STATIC.PARTICLE_SPACING if spacing is None else spacing)

    def classify_water(self, positions, groups, u, spacing=None):
        """Original particle partitions using actual full-R rigid containment."""
        points, labels = np.asarray(positions, dtype=float), np.asarray(groups, dtype=np.int32)
        spacing = _STATIC.PARTICLE_SPACING if spacing is None else float(spacing)
        if (points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),)
                or not np.isfinite(points).all() or np.any((labels != 0) & (labels != 1))
                or not math.isfinite(spacing) or spacing <= 0):
            raise ValueError("Finite Nx3 points, fixed 0/1 labels and positive spacing are required.")
        delta = points - np.asarray(_STATIC.BEAKER_CENTER)
        beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= _STATIC.BEAKER_RADIUS ** 2)
                  & (delta[:, 2] >= 0) & (delta[:, 2] <= _STATIC.BEAKER_HEIGHT))
        transform = self.tube_matrix(u)
        tube = _STATIC._cavity_mask((points - transform[:3, 3]) @ transform[:3, :3])
        free = ~(beaker | tube)
        airborne = free & (points[:, 2] > _STATIC.TABLE_TOP_Z + spacing)
        spilled, initial_tube = free & ~airborne, labels == 1
        source_count = int(np.count_nonzero(initial_tube))
        received = int(np.count_nonzero(initial_tube & beaker))
        quantum = spacing ** 3 * 1e6
        return {"time": self._clip(u), "phase": self.phase(u), "total": len(points),
                "beaker": int(np.count_nonzero(beaker)), "tube": int(np.count_nonzero(tube)),
                "airborne": int(np.count_nonzero(airborne)), "spilled": int(np.count_nonzero(spilled)),
                "below_table": int(np.count_nonzero(free & (points[:, 2] < _STATIC.TABLE_TOP_Z - spacing))),
                "initial_tube": source_count, "tube_to_beaker": received,
                "tube_remaining": int(np.count_nonzero(initial_tube & tube)),
                "tube_airborne": int(np.count_nonzero(initial_tube & airborne)),
                "tube_spilled": int(np.count_nonzero(initial_tube & spilled)),
                "tube_transfer_fraction": received / source_count if source_count else 0.,
                "total_particle_volume_ml": len(points) * quantum, "initial_tube_volume_ml": source_count * quantum,
                "beaker_volume_ml": int(np.count_nonzero(beaker)) * quantum,
                "tube_volume_ml": int(np.count_nonzero(tube)) * quantum}


def load_geometry(cache_file, expected_sha256):
    return PivotGeometry(cache_file, expected_sha256)


_DEFAULT = load_geometry(DEFAULT_CACHE_FILE, DEFAULT_CACHE_SHA256)


def __getattr__(name):
    return getattr(_DEFAULT, name)
