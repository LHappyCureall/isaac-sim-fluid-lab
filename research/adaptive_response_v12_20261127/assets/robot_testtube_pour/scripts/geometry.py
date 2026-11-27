"""Pure NumPy geometry and prescribed motion for the robot pouring example.

No Kit, physics-engine, or UI imports. Lengths are meters, time is seconds,
Z is up. Meshes have outward-oriented triangles around solid glass volumes.
This module never clamps or moves simulated liquid.

Pose contracts:
  mouth_pose(t) -> (mouth_xyz, tilt_radians)
  tube_pose(t)  -> (tube_inner_bottom_xyz, quaternion_xyzw)
  hand_pose(t)  -> (panda_hand_xyz, quaternion_wxyz)
"""

import math

import numpy as np


DURATION = 9.4
BEAKER_CENTER = (0.48, 0.10, 0.004)
BEAKER_RADIUS = 0.065
BEAKER_HEIGHT = 0.14
BEAKER_THICKNESS = 0.004
BEAKER_WATER_DEPTH = 0.014
TUBE_RADIUS = 0.0135
TUBE_LENGTH = 0.18
TUBE_THICKNESS = 0.003
TUBE_FILL_HEIGHT = 0.12
PARTICLE_SPACING = 0.0035
TCP_BELOW_MOUTH = 0.09
PANDA_HAND_TO_TCP = 0.1034
TABLE_TOP_Z = 0.0

# End-of-phase time, mouth world position, tube tilt about world Y.
MOTION_KNOTS = (
    (0.0, (0.35, -0.20, 0.44), 0.0),
    (0.7, (0.35, -0.20, 0.48), 0.0),
    (2.2, (0.43, 0.10, 0.42), 0.0),
    (3.2, (0.43, 0.10, 0.32), 65.0),
    (4.2, (0.43, 0.10, 0.205), 125.0),
    (6.6, (0.43, 0.10, 0.205), 125.0),
    (7.8, (0.43, 0.10, 0.42), 0.0),
    (DURATION, (0.35, -0.20, 0.44), 0.0),
)
PHASE_NAMES = ("lift", "transfer", "pretilt", "pour", "hold", "return", "retreat")


def _validate_dimensions(radius, height, thickness, segments):
    if not all(math.isfinite(float(x)) and float(x) > 0.0 for x in (radius, height, thickness)):
        raise ValueError("Radius, length/height, and wall thickness must be finite and positive.")
    if int(segments) != segments or segments < 3:
        raise ValueError("segments must be an integer of at least three.")


def rounded_testtube_mesh(r=TUBE_RADIUS, L=TUBE_LENGTH, thickness=TUBE_THICKNESS, segments=64):
    """Return local-space hollow test-tube glass (points, flat triangle indices).

    The inner hemisphere is centered at z=r and has its pole at z=0. The
    outer hemisphere shares that center and has radius r+thickness, so its
    pole is z=-thickness. Both cylindrical walls end at z=L, joined by an
    upward-facing annular lip. The open cavity is not capped by a disk.
    """
    _validate_dimensions(r, L, thickness, segments)
    if L <= r:
        raise ValueError("Tube length must exceed the inner hemisphere radius.")
    segments = int(segments)
    hemisphere_rings = 12
    points = []
    faces = []
    angles = np.arange(segments, dtype=np.float64) * (2.0 * math.pi / segments)

    def ring(radius, z):
        first = len(points)
        points.extend((float(radius * math.cos(a)), float(radius * math.sin(a)), float(z)) for a in angles)
        return list(range(first, first + segments))

    def quad(a, b, c, d):
        faces.extend((a, b, c, a, c, d))

    def wall(shell_radius, inward):
        pole = len(points)
        points.append((0.0, 0.0, float(r - shell_radius)))
        rings = []
        for k in range(1, hemisphere_rings + 1):
            latitude = -0.5 * math.pi + k * (0.5 * math.pi / hemisphere_rings)
            rings.append(ring(shell_radius * math.cos(latitude), r + shell_radius * math.sin(latitude)))
        rings.append(ring(shell_radius, L))
        for i in range(segments):
            j = (i + 1) % segments
            if inward:
                faces.extend((pole, rings[0][i], rings[0][j]))
            else:
                faces.extend((pole, rings[0][j], rings[0][i]))
        for lower, upper in zip(rings[:-1], rings[1:]):
            for i in range(segments):
                j = (i + 1) % segments
                if inward:
                    quad(lower[j], lower[i], upper[i], upper[j])
                else:
                    quad(lower[i], lower[j], upper[j], upper[i])
        return rings[-1]

    inner_rim = wall(float(r), True)
    outer_rim = wall(float(r + thickness), False)
    for i in range(segments):
        j = (i + 1) % segments
        quad(inner_rim[i], outer_rim[i], outer_rim[j], inner_rim[j])
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def closed_beaker_mesh(
    radius=BEAKER_RADIUS,
    height=BEAKER_HEIGHT,
    thickness=BEAKER_THICKNESS,
    segments=96,
    center=BEAKER_CENTER,
):
    """Return a closed solid glass beaker mesh with world-space points.

    center is the center of the INNER floor. Outer floor is one thickness
    lower; the upper opening remains open, bounded by an annular glass rim.
    """
    _validate_dimensions(radius, height, thickness, segments)
    segments = int(segments)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain three finite coordinates.")
    points = []
    faces = []

    def ring(r, z):
        first = len(points)
        for angle in np.arange(segments) * (2.0 * math.pi / segments):
            points.append((float(r * math.cos(angle)), float(r * math.sin(angle)), float(z)))
        return list(range(first, first + segments))

    def quad(a, b, c, d):
        faces.extend((a, b, c, a, c, d))

    inner_floor = ring(radius, 0.0)
    inner_rim = ring(radius, height)
    outer_floor = ring(radius + thickness, -thickness)
    outer_rim = ring(radius + thickness, height)
    inner_center = len(points)
    points.append((0.0, 0.0, 0.0))
    outer_center = len(points)
    points.append((0.0, 0.0, -float(thickness)))
    for i in range(segments):
        j = (i + 1) % segments
        quad(inner_floor[j], inner_floor[i], inner_rim[i], inner_rim[j])
        quad(outer_floor[i], outer_floor[j], outer_rim[j], outer_rim[i])
        quad(inner_rim[i], outer_rim[i], outer_rim[j], inner_rim[j])
        faces.extend((inner_center, inner_floor[i], inner_floor[j]))
        faces.extend((outer_center, outer_floor[j], outer_floor[i]))
    return (np.asarray(points, dtype=np.float64) + center).astype(np.float32), np.asarray(faces, dtype=np.int32)


def _time(t):
    t = float(t)
    if not math.isfinite(t):
        raise ValueError("Time must be finite.")
    return min(max(t, 0.0), DURATION)


def smootherstep(u):
    """Quintic interpolation with zero velocity/acceleration at both ends."""
    u = min(max(float(u), 0.0), 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def phase(t):
    """Return a stable English phase label for status/UI/validation."""
    raw_time = float(t)
    if not math.isfinite(raw_time):
        raise ValueError("Time must be finite.")
    if raw_time <= 0.0:
        return "ready"
    if raw_time >= DURATION:
        return "complete"
    for index, knot in enumerate(MOTION_KNOTS[1:]):
        if raw_time < knot[0]:
            return PHASE_NAMES[index]
    return "complete"


def mouth_pose(t):
    """Return (mouth world XYZ, tilt radians about Y), both eased per phase."""
    t = _time(t)
    if t >= DURATION:
        return np.asarray(MOTION_KNOTS[-1][1], dtype=np.float64), 0.0
    for start, end in zip(MOTION_KNOTS[:-1], MOTION_KNOTS[1:]):
        if t <= end[0]:
            u = smootherstep((t - start[0]) / (end[0] - start[0]))
            a = np.asarray(start[1], dtype=np.float64)
            b = np.asarray(end[1], dtype=np.float64)
            tilt = math.radians(start[2] + u * (end[2] - start[2]))
            return a + u * (b - a), tilt
    raise RuntimeError("Motion phase lookup failed.")


def rotation_y(angle):
    """Active rotation matrix: local +Z leans toward world +X for +angle."""
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)


def tube_rotation(t):
    return rotation_y(mouth_pose(t)[1])


def tube_pose(t):
    """Return tube local inner-bottom origin and an XYZW quaternion."""
    mouth, angle = mouth_pose(t)
    origin = mouth - rotation_y(angle) @ np.asarray((0.0, 0.0, TUBE_LENGTH))
    return origin, np.asarray((0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)))


def hand_pose(t):
    """Return panda_hand world origin and a WXYZ quaternion.

    Tool Z points sideways into the upright tube, avoiding a palm across
    its opening. panda_hand-to-TCP is +0.1034m along hand-local Z.
    """
    mouth, angle = mouth_pose(t)
    rt = rotation_y(angle)
    hand_angle = angle + 0.5 * math.pi
    rh = rotation_y(hand_angle)
    tcp = mouth - rt @ np.asarray((0.0, 0.0, TCP_BELOW_MOUTH))
    hand = tcp - rh @ np.asarray((0.0, 0.0, PANDA_HAND_TO_TCP))
    quat_wxyz = np.asarray((math.cos(0.5 * hand_angle), 0.0, math.sin(0.5 * hand_angle), 0.0))
    return hand, quat_wxyz


def _cavity_mask(local_points, radius=TUBE_RADIUS, length=TUBE_LENGTH):
    z = local_points[:, 2]
    radial_squared = np.sum(local_points[:, :2] ** 2, axis=1)
    cap_or_wall = np.where(z < radius, radial_squared + (z - radius) ** 2 <= radius**2, radial_squared <= radius**2)
    return (z >= 0.0) & (z <= length) & cap_or_wall


def initial_water_points(spacing=PARTICLE_SPACING):
    """Return (world points float32, labels int32), 0=beaker, 1=test tube.

    A half-spacing clearance is used against container walls and floors.
    Each point represents spacing**3 cubic meters of liquid at unit fill.
    """
    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0.0 or spacing >= TUBE_RADIUS:
        raise ValueError("spacing must be finite, positive, and smaller than the tube radius.")
    margin = 0.5 * spacing

    def cylinder_grid(radius, z0, z1):
        n = int(math.ceil(radius / spacing))
        axis = np.arange(-n, n + 1, dtype=np.float64) * spacing
        heights = np.arange(z0, z1 + 0.01 * spacing, spacing)
        grid = np.stack(np.meshgrid(axis, axis, heights, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid[np.sum(grid[:, :2] ** 2, axis=1) <= radius**2]

    beaker = cylinder_grid(BEAKER_RADIUS - margin, margin, BEAKER_WATER_DEPTH - margin)
    beaker += np.asarray(BEAKER_CENTER)
    tube = cylinder_grid(TUBE_RADIUS - margin, margin, TUBE_FILL_HEIGHT - margin)
    lower_cap = tube[:, 2] < TUBE_RADIUS
    cap_inside = np.sum(tube[:, :2] ** 2, axis=1) + (tube[:, 2] - TUBE_RADIUS) ** 2 <= (TUBE_RADIUS - margin) ** 2
    tube = tube[(~lower_cap) | cap_inside]
    origin, _ = tube_pose(0.0)
    tube = tube @ tube_rotation(0.0).T + origin
    points = np.concatenate((beaker, tube), axis=0).astype(np.float32)
    labels = np.concatenate((np.zeros(len(beaker), dtype=np.int32), np.ones(len(tube), dtype=np.int32)))
    return points, labels


def classify_water(positions, groups, t, spacing=PARTICLE_SPACING):
    """Return JSON-compatible containment/transfer metrics; never alter data.

    "airborne" means outside both containers and >one spacing above the
    tabletop. "spilled" is outside both containers on/below that level;
    "below_table" is a diagnostic subset of spilled. Exact cavity tests
    use particle centers rather than rendered surface bounds.
    """
    points = np.asarray(positions, dtype=np.float64)
    labels = np.asarray(groups, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError("positions must be (N,3) and groups must be (N,).")
    if not np.all(np.isfinite(points)):
        raise ValueError("Particle positions must be finite.")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("groups must use 0 for initial beaker water and 1 for initial tube water.")
    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive.")
    center = np.asarray(BEAKER_CENTER)
    delta = points - center
    beaker = (np.sum(delta[:, :2] ** 2, axis=1) <= BEAKER_RADIUS**2) & (delta[:, 2] >= 0.0) & (delta[:, 2] <= BEAKER_HEIGHT)
    origin, _ = tube_pose(t)
    tube = _cavity_mask((points - origin) @ tube_rotation(t))
    # Beaker and tube interiors are disjoint on the prescribed trajectory.
    free = ~(beaker | tube)
    airborne = free & (points[:, 2] > TABLE_TOP_Z + spacing)
    spilled = free & ~airborne
    below_table = free & (points[:, 2] < TABLE_TOP_Z - spacing)
    initial_tube = labels == 1
    count_tube = int(np.count_nonzero(initial_tube))
    tube_received = int(np.count_nonzero(initial_tube & beaker))
    particle_ml = spacing**3 * 1.0e6
    return {
        "time": float(_time(t)),
        "phase": phase(t),
        "total": int(len(points)),
        "beaker": int(np.count_nonzero(beaker)),
        "tube": int(np.count_nonzero(tube)),
        "airborne": int(np.count_nonzero(airborne)),
        "spilled": int(np.count_nonzero(spilled)),
        "below_table": int(np.count_nonzero(below_table)),
        "initial_tube": count_tube,
        "tube_to_beaker": tube_received,
        "tube_remaining": int(np.count_nonzero(initial_tube & tube)),
        "tube_airborne": int(np.count_nonzero(initial_tube & airborne)),
        "tube_spilled": int(np.count_nonzero(initial_tube & spilled)),
        "tube_transfer_fraction": float(tube_received / count_tube) if count_tube else 0.0,
        "total_particle_volume_ml": float(len(points) * particle_ml),
        "initial_tube_volume_ml": float(count_tube * particle_ml),
        "beaker_volume_ml": float(np.count_nonzero(beaker) * particle_ml),
        "tube_volume_ml": float(np.count_nonzero(tube) * particle_ml),
    }
