"""Franka side-grips a filled test tube and pours simulated liquid into a beaker.

Isaac Sim: Window > Script Editor > open this file > Run, then press Play.
Pause freezes the example; Stop restores its original water and robot pose.
The arm follows validated kinematic IK poses. The tube is a prescribed
kinematic collider; the liquid starts as one fixed set of particles.
This prototype does not simulate contact-force grasping or fluid torque feedback.
"""

import asyncio
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from lab_paths import PROJECT_ROOT

import numpy as np

# ---- User options -----------------------------------------------------------
TABLE_STYLE = "white_matte"       # pale_oak / white_marble / gray_concrete / white_matte
RENDER_MODE = "RealTimePathTracing"   # preview; "PathTracing" for paused stills
PARTICLE_SPACING = 0.0035      # meters; smaller = finer liquid and more work
COLLISION_GRID_SIZE = 0.007   # native collision/surface helper grid, meters
SURFACE_VOXEL_SIZE = 0.0025
SURFACE_MAX_GRID_CELLS = 4_000_000  # display extraction capacity; no fluid physics change
SIMULATION_FPS = 60.0         # fixed simulation time step, not a frame-rate promise
MOTION_TIME_SCALE = 2.0       # physical source timing; keep 2.0 for the saved cache
STOP_AFTER_UPRIGHT = True    # finish upright; False also returns to the starting pose
POST_POUR_HOLD_SECONDS = 0.25  # replay/video dwell after pouring (was 4.8 s)
UPRIGHT_PLAYBACK_SPEED = 2.0  # replay/video upright motion: 2x = 1.2 s
RETURN_PLAYBACK_SPEED = 1.6   # replay/video return to start, if enabled
SUBSTEPS = 4
SURFACE_EVERY_N_STEPS = 2     # 30 surface reconstructions per simulated second
SHOW_CONTROL_PANEL = True
LIQUID_MODE = "simulate"        # replay = saved physical simulation; simulate = solve live
ROBOT_BASE = (0.0, -0.40, 0.0)
ROOT = "/World/RobotPourDemo"
OUTPUT = PROJECT_ROOT / "droplet_out" / "robot_testtube_pour"
INSTALL = PROJECT_ROOT
CACHE_FILE = INSTALL / "assets" / "robot_testtube_pour" / "pour_cache.npz"
VIDEO_SECONDS = 10.0         # MP4 playback duration, independent of render time
VIDEO_FPS = 30               # 10 seconds x 30 FPS = exactly 300 encoded frames
VIDEO_RESOLUTION = (1280, 720)


def _demo_end_time(geometry):
    return float(geometry.MOTION_KNOTS[-2][0] if STOP_AFTER_UPRIGHT else geometry.DURATION)


def _replay_schedule(geometry):
    """Map presentation seconds to the unchanged physical cache and IK poses."""
    if (not math.isfinite(POST_POUR_HOLD_SECONDS) or POST_POUR_HOLD_SECONDS <= 0
            or not math.isfinite(UPRIGHT_PLAYBACK_SPEED) or UPRIGHT_PLAYBACK_SPEED <= 0
            or not math.isfinite(RETURN_PLAYBACK_SPEED) or RETURN_PLAYBACK_SPEED <= 0):
        raise ValueError("Replay hold duration and motion speeds must be finite and positive.")
    pour_end, hold_end, upright = (float(geometry.MOTION_KNOTS[i][0]) for i in (4, 5, 6))
    source = [0.0, pour_end, hold_end, upright]
    playback = [0.0, pour_end, pour_end + POST_POUR_HOLD_SECONDS]
    playback.append(playback[-1] + (upright - hold_end) / UPRIGHT_PLAYBACK_SPEED)
    if not STOP_AFTER_UPRIGHT:
        source.append(float(geometry.DURATION))
        playback.append(playback[-1] + (source[-1] - upright) / RETURN_PLAYBACK_SPEED)
    return np.asarray(playback), np.asarray(source)


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Read source directly: Script Editor hot reload must not reuse a stale pyc.
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _load_parts():
    """Load the example's reusable geometry, robot, and liquid implementation."""
    directory = INSTALL / "assets" / "robot_testtube_pour" / "scripts"
    return (
        _load_file("robot_pour_geometry", directory / "geometry.py"),
        _load_file("robot_pour_robot", directory / "robot.py"),
        _load_file("robot_pour_fluid", directory / "liquid.py"),
    )


def _enable_dependencies():
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in ("isaacsim.pip.newton", "isaacsim.robot.schema",
                      "isaacsim.robot.poser", "isaacsim.asset.importer.urdf"):
        manager.set_extension_enabled_immediate(extension, True)


class RecordedFluid:
    """Replay recorded particle states, retaining live surface reconstruction.

    These are states computed by the liquid solver, with fixed particle IDs.
    Interpolation makes rendering independent of the expensive solver clock.
    Set LIQUID_MODE="simulate" above to run the solver again instead.
    """

    def __init__(self, live, path, geometry):
        with np.load(path, allow_pickle=False) as archive:
            self.frames = archive["positions"].astype(np.float32, copy=True)
            self.times = archive["times"].astype(float, copy=True)
            labels = archive["labels"]
            self.cache_metadata = json.loads(str(archive["metadata"].item()))
        if (self.frames.shape[1:] != (live.particle_count, 3)
                or len(self.frames) != len(self.times) or len(self.times) < 2
                or not np.isfinite(self.frames).all() or not np.isfinite(self.times).all()
                or self.times[0] != 0.0 or np.any(np.diff(self.times) <= 0)
                or abs(self.times[-1] - geometry.DURATION) > 1e-5
                or not np.array_equal(labels, live.labels)
                or abs(self.cache_metadata["spacing"] - PARTICLE_SPACING) > 1e-9
                or abs(self.cache_metadata["motion_time_scale"] - MOTION_TIME_SCALE) > 1e-9
                or not self.cache_metadata.get("validation_success", False)
                or not np.allclose(self.frames[0], live.positions(), atol=1e-7)):
            raise ValueError("Saved liquid does not match these physics settings. "
                             "Use LIQUID_MODE='simulate' to run changed settings.")
        expected_geometry = {
            "tube_radius": geometry.TUBE_RADIUS, "tube_length": geometry.TUBE_LENGTH,
            "tube_thickness": geometry.TUBE_THICKNESS, "beaker_radius": geometry.BEAKER_RADIUS,
            "beaker_height": geometry.BEAKER_HEIGHT, "beaker_center": list(geometry.BEAKER_CENTER),
            "beaker_thickness": geometry.BEAKER_THICKNESS}
        if self.cache_metadata.get("geometry") != expected_geometry:
            raise ValueError("Saved liquid geometry changed. Use LIQUID_MODE='simulate'.")
        self.live = live
        self._frame_index = 0
        self._time = 0.0
        self.mode = "recorded physical simulation: " + live.mode

    def __getattr__(self, name):
        return getattr(self.live, name)

    @property
    def time(self):
        return self._time

    def step_frame(self):
        self.seek((self._frame_index + 1) * self.live.dt)

    def seek(self, t):
        """Sample a saved physical state without advancing the solver."""
        t = float(np.clip(t, 0.0, self.times[-1]))
        self._time = t
        self._frame_index = int(round(t / self.live.dt))
        index = int(np.clip(np.searchsorted(self.times, t, side="right") - 1,
                            0, len(self.times) - 2))
        interval = self.times[index + 1] - self.times[index]
        alpha = float(np.clip((t - self.times[index]) / interval, 0.0, 1.0))
        a, b = self.frames[index:index + 2]
        self.live.state_0.particle_q.assign((1 - alpha) * a + alpha * b)
        self.live.state_0.particle_qd.assign((b - a) / interval)
        self.live._set_tube_state(t)

    def reset(self):
        self.live.reset()
        self._frame_index = 0
        self._time = 0.0

    def positions(self):
        return self.live.positions()

    def extract_surface(self):
        return self.live.extract_surface()


def _pose_matrix(position, quaternion_xyzw):
    from pxr import Gf
    q = np.asarray(quaternion_xyzw, dtype=float)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
    matrix.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
    return matrix


def _make_mesh(stage, path, vertices, faces, material=None):
    from pxr import UsdGeom, UsdShade, Vt
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    normals = np.zeros_like(points)
    cross = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                     points[triangles[:, 2]] - points[triangles[:, 0]])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _set_camera(stage, name, eye, target, focal):
    from pxr import Gf, UsdGeom
    camera = UsdGeom.Camera.Define(stage, ROOT + "/" + name)
    camera.CreateFocalLengthAttr(float(focal))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    matrix = Gf.Matrix4d()
    matrix.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera.GetPrim()).MakeMatrixXform().Set(matrix.GetInverse())
    return str(camera.GetPath())


def _build_visuals(stage, geometry, looks):
    import carb
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    looks._setup_scene_lighting(stage)
    looks._apply_table_material(stage, TABLE_STYLE)
    table = UsdGeom.Mesh.Get(stage, "/World/NewtonTable")
    table.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-.35, -.65, 0), Gf.Vec3f(1.05, -.65, 0),
        Gf.Vec3f(1.05, .65, 0), Gf.Vec3f(-.35, .65, 0)]))

    beaker_data = geometry.closed_beaker_mesh()
    beaker_visual = beaker_data[0].copy()
    outer_floor = geometry.BEAKER_CENTER[2] - geometry.BEAKER_THICKNESS
    # A tiny optical gap resolves the coincident glass exit face/tabletop.
    beaker_visual[np.isclose(beaker_visual[:, 2], outer_floor, atol=1e-7), 2] += .0003
    beaker = _make_mesh(stage, ROOT + "/Beaker", beaker_visual, beaker_data[1])
    # Keep a flat floor/rim and smooth cylindrical walls, with hard creases.
    beaker_normals = []
    for index, face in enumerate(beaker_data[1].reshape(-1, 3)):
        kind = index % 8
        for vertex in beaker_visual[face]:
            if kind < 4:
                radial = vertex[:2] - np.asarray(geometry.BEAKER_CENTER[:2])
                radial /= np.linalg.norm(radial)
                sign = -1 if kind < 2 else 1
                beaker_normals.append((sign * radial[0], sign * radial[1], 0.0))
            else:
                beaker_normals.append((0.0, 0.0, -1.0 if kind == 7 else 1.0))
    beaker.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(beaker_normals, dtype=np.float32)))
    beaker.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    looks._apply_glass_material(stage, beaker)
    tube_root = UsdGeom.Xform.Define(stage, ROOT + "/TestTube")
    tube_op = tube_root.MakeMatrixXform()
    tube_data = geometry.rounded_testtube_mesh()
    tube = _make_mesh(stage, ROOT + "/TestTube/Glass", *tube_data)
    looks._apply_glass_material(stage, tube)
    water = _make_mesh(stage, ROOT + "/Water", np.empty((0, 3)), np.empty(0, dtype=np.int32))
    water.CreateDoubleSidedAttr(True)
    looks._apply_water_material(stage, water)
    # Actual transparent objects still cast shadows; no all-shadow disable.
    for item in (beaker, tube, water):
        UsdGeom.PrimvarsAPI(item.GetPrim()).CreatePrimvar(
            "enableFastRefractionShadow", Sdf.ValueTypeNames.Bool).Set(True)

    ink = looks.crown._omnipbr(stage, ROOT + "/Looks/Graduation", (.13, .20, .25), 1, .65, .2)
    marks = UsdGeom.BasisCurves.Define(stage, ROOT + "/Graduations")
    pts = []
    center = np.asarray(geometry.BEAKER_CENTER)
    radius = geometry.BEAKER_RADIUS + geometry.BEAKER_THICKNESS + .0005
    for level in np.arange(.025, geometry.BEAKER_HEIGHT, .02):
        for x in (-.010, .010):
            pts.append(Gf.Vec3f(center[0] + x, center[1] - math.sqrt(radius * radius - x * x),
                               center[2] + level))
    marks.CreateTypeAttr(UsdGeom.Tokens.linear)
    marks.CreateCurveVertexCountsAttr(Vt.IntArray([2] * (len(pts) // 2)))
    marks.CreatePointsAttr(Vt.Vec3fArray(pts))
    marks.CreateWidthsAttr(Vt.FloatArray([.0007]))
    marks.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    UsdShade.MaterialBindingAPI.Apply(marks.GetPrim()).Bind(ink)

    wide = _set_camera(stage, "Overview", (1.65, -1.75, 1.25), (.15, -.16, .43), 24)
    close = _set_camera(stage, "PourClose", (1.05, -.71, .54), (.45, .10, .195), 24)
    settings = carb.settings.get_settings()
    settings.set_string("/rtx/rendermode", RENDER_MODE)
    settings.set_int("/rtx/rtpt/maxBounces", 12)
    settings.set_int("/rtx/rtpt/maxSpecularAndTransmissionBounces", 12)
    settings.set_int("/rtx/pathtracing/maxBounces", 10)
    settings.set_int("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 10)
    settings.set_bool("/app/viewport/grid/enabled", False)
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = wide
    return water, tube_op, tube_data, beaker_data, {"overview": wide, "close": close}


class PourDemo:
    def __init__(self, stage, geometry, rig, fluid, water, tube_op, cameras):
        import omni.kit.app
        import omni.usd
        self.stage, self.geometry, self.rig, self.fluid = stage, geometry, rig, fluid
        self._scene_epoch = omni.kit.app._water_demo_epoch
        self._stage_id = omni.usd.get_context().get_stage_id()
        self._closed = False
        self.end_time = _demo_end_time(geometry)
        self._playback_knots, self._source_knots = _replay_schedule(geometry)
        self.playback_end_time = (float(self._playback_knots[-1])
                                  if isinstance(fluid, RecordedFluid) else self.end_time)
        self.water, self.tube_op, self.cameras = water, tube_op, cameras
        self.busy = False
        self.step_count = 0
        self.history = []
        self.phase_name = "ready"
        self.max_grasp_error = 0.0
        self.step_times = []
        self.surface_times = []
        self.initial_mass = fluid.total_mass
        self.initial_count = fluid.particle_count
        self._finished = False
        self.sync_visuals(force_surface=True)
        self.sample()

    def _owns_scene(self):
        import omni.kit.app
        import omni.usd
        return (not self._closed
                and getattr(omni.kit.app, "_water_demo_owner", None) == "robot_pour"
                and getattr(omni.kit.app, "_water_demo_epoch", None) == self._scene_epoch
                and omni.usd.get_context().get_stage_id() == self._stage_id
                and self.stage.GetPrimAtPath(ROOT).IsValid())

    def sync_visuals(self, force_surface=False):
        from pxr import Vt
        self.rig.update(self.fluid.time)
        # Read the collider pose back from the simulation state.
        pose = self.fluid.state_0.body_q.numpy()[self.fluid.tube_body]
        self.tube_op.Set(_pose_matrix(pose[:3], pose[3:]))
        if force_surface or self.step_count % SURFACE_EVERY_N_STEPS == 0:
            start = time.perf_counter()
            points, faces, normals = self.fluid.extract_surface()
            self.water.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            self.water.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(faces) // 3)))
            self.water.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
            self.water.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
            self.surface_times.append(time.perf_counter() - start)

    def sample(self):
        from pxr import UsdGeom
        positions = self.fluid.positions()
        metrics = self.geometry.classify_water(positions, self.fluid.labels,
                                               self.fluid.time, PARTICLE_SPACING)
        if len(positions) != self.initial_count or not np.isfinite(positions).all():
            raise RuntimeError("Fluid particle count changed or a non-finite position appeared.")
        actual_hand = UsdGeom.Xformable(self.rig.hand).ComputeLocalToWorldTransform(0)
        actual_tcp = actual_hand.Transform((0.0, 0.0, self.geometry.PANDA_HAND_TO_TCP))
        tube_position, _ = self.geometry.tube_pose(self.fluid.time)
        tube_grasp = tube_position + self.geometry.tube_rotation(self.fluid.time) @ np.array(
            [0.0, 0.0, self.geometry.TUBE_LENGTH - self.geometry.TCP_BELOW_MOUTH])
        grasp_error = float(np.linalg.norm(np.array(actual_tcp) - tube_grasp))
        self.max_grasp_error = max(self.max_grasp_error, grasp_error)
        metrics["grasp_position_error_m"] = grasp_error
        metrics["mass_kg"] = self.fluid.total_mass
        original_tube = positions[self.fluid.labels == 1]
        local_tube = (original_tube - tube_position) @ self.geometry.tube_rotation(self.fluid.time)
        metrics["tube_particle_local_z_span_m"] = float(np.ptp(local_tube[:, 2]))
        self.history.append(metrics)
        label = getattr(self, "status_label", None)
        if label is not None:
            elapsed = (float(np.interp(self.fluid.time, self._source_knots, self._playback_knots))
                       if isinstance(self.fluid, RecordedFluid) else self.fluid.time)
            label.text = (f"{elapsed:.1f} / {self.playback_end_time:.1f} s   "
                          f"{self.phase_name}\nReceived: "
                          f"{100 * metrics['tube_transfer_fraction']:.1f}% of tube water")
        return metrics

    def advance(self, force_surface=False):
        if self.fluid.time >= self.end_time - 1e-8:
            return False
        start = time.perf_counter()
        self.fluid.step_frame()
        self.step_times.append(time.perf_counter() - start)
        self.step_count += 1
        self.sync_visuals(force_surface)
        phase_name = ("complete" if self.fluid.time >= self.end_time - 1e-8
                      else self.geometry.phase(self.fluid.time))
        if phase_name != self.phase_name or self.step_count % int(SIMULATION_FPS) == 0:
            self.phase_name = phase_name
            metrics = self.sample()
            print(f"Robot pour t={self.fluid.time:.2f}s [{phase_name}] "
                  f"received={100 * metrics['tube_transfer_fraction']:.1f}% "
                  f"remaining={metrics['tube_remaining']} particles", flush=True)
        return True

    def playback_source_time(self, playback_time):
        """Sample robot, tube and liquid together, including a faster ending."""
        return float(np.interp(float(playback_time), self._playback_knots, self._source_knots))

    def video_source_time(self, video_time, duration=VIDEO_SECONDS):
        """Fit the same replay schedule to an exact MP4 duration."""
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration must be finite and positive.")
        return self.playback_source_time(float(video_time) / duration * self._playback_knots[-1])

    def seek_recorded(self, source_time):
        if not self._owns_scene():
            raise asyncio.CancelledError("The active water example changed.")
        if not isinstance(self.fluid, RecordedFluid):
            raise RuntimeError("MP4 export requires LIQUID_MODE='replay'.")
        self.fluid.seek(source_time)
        self.step_count = int(round(source_time * SIMULATION_FPS))
        self.sync_visuals(force_surface=True)

    def export_video(self):
        if not self._owns_scene():
            return None
        previous = getattr(self, "export_task", None)
        if previous is not None and not previous.done():
            return previous

        async def job():
            try:
                video = _load_file("robot_pour_video", INSTALL / "assets" /
                                   "robot_testtube_pour" / "scripts" / "video.py")
                path = OUTPUT / f"robot_pour_{VIDEO_SECONDS:g}s.mp4"
                result = await video.export_demo_mp4(self, path, VIDEO_SECONDS, VIDEO_FPS,
                                                    *VIDEO_RESOLUTION)
                if not self._owns_scene():
                    raise asyncio.CancelledError("The pouring scene was replaced.")
                if getattr(self, "status_label", None) is not None:
                    self.status_label.text = f"Saved {VIDEO_SECONDS:g}s MP4\n{path.name}"
                print(f"MP4 saved: {path}", flush=True)
                return result
            except Exception:
                if not self._owns_scene():
                    raise asyncio.CancelledError("The pouring scene was replaced.") from None
                import traceback
                traceback.print_exc()
                if getattr(self, "status_label", None) is not None:
                    self.status_label.text = "Export failed - see Script Editor log"
                raise

        self.export_task = asyncio.ensure_future(job())
        return self.export_task

    def reset(self):
        if not self._owns_scene():
            return
        self.fluid.reset()
        self.step_count = 0
        self.history = []
        self.phase_name = "ready"
        self.max_grasp_error = 0.0
        self.step_times = []
        self.surface_times = []
        self._finished = False
        self.sync_visuals(force_surface=True)
        self.sample()
        print("Robot pour reset. Press Play to start.", flush=True)

    def report(self):
        final = self.sample()
        mass_conserved = self.fluid.total_mass == self.initial_mass
        upright = [item for item in self.history
                   if .2 * MOTION_TIME_SCALE <= item["time"] <= 2.2 * MOTION_TIME_SCALE]
        initial_span = self.history[0]["tube_particle_local_z_span_m"]
        upright_height = min((item["tube_particle_local_z_span_m"] / initial_span
                              for item in upright), default=0.0)
        upright_retention = min((item["tube_remaining"] / item["initial_tube"]
                                 for item in upright), default=0.0)
        static_hold_passed = upright_height >= .65 and upright_retention >= .95
        joint_speeds_passed = self.rig.metrics["max_joint_velocity_limit_ratio"] <= 1.0
        passed = (self.rig.metrics["success"] and mass_conserved
                  and static_hold_passed and joint_speeds_passed
                  and final["tube_transfer_fraction"] >= .75
                  and final["below_table"] == 0 and self.max_grasp_error < .003)
        return {
            "success": bool(passed),
            "mode": "kinematic_Franka_and_prescribed_tube_with_fixed_particle_liquid",
            "fluid_solver": getattr(self.fluid, "mode", type(self.fluid).__name__),
            "playback_mode": LIQUID_MODE,
            "playback_duration_seconds": self.playback_end_time,
            "joint_velocity_validation_time_basis": "physical_source_seconds",
            "physics_cache_validation": getattr(self.fluid, "cache_metadata", None),
            "contact_force_grasp_simulated": False,
            "particle_count": self.initial_count,
            "total_mass_kg": self.initial_mass,
            "mass_conserved": bool(mass_conserved),
            "upright_hold_passed": bool(static_hold_passed),
            "joint_speeds_within_urdf_limits": bool(joint_speeds_passed),
            "min_upright_height_fraction": upright_height,
            "min_upright_particle_retention": upright_retention,
            "max_grasp_position_error_m": self.max_grasp_error,
            "mean_solver_frame_seconds": float(np.mean(self.step_times)) if self.step_times else 0,
            "mean_surface_seconds": float(np.mean(self.surface_times)) if self.surface_times else 0,
            "robot": self.rig.metrics, "final": final, "history": self.history,
        }

    def save_report(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        result = self.report()
        (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def attach(self):
        import omni.kit.app
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0.0)
        # A short guard interval lets our callback pause on the upright final pose
        # before the native timeline can emit an automatic STOP/reset.
        timeline.set_end_time(self.playback_end_time + 0.1)
        timeline.set_time_codes_per_second(SIMULATION_FPS)
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(self.playback_end_time * SIMULATION_FPS)
        self.stage.SetTimeCodesPerSecond(SIMULATION_FPS)
        self.stage.SetFramesPerSecond(SIMULATION_FPS)
        timeline.set_looping(False)

        def on_update(_event):
            if not self._owns_scene() or self.busy or not timeline.is_playing():
                return
            self.busy = True
            try:
                if isinstance(self.fluid, RecordedFluid):
                    # Movie Capture may render several app updates at the same
                    # time code. Sample that time instead of advancing per render.
                    target = self.playback_source_time(timeline.get_current_time())
                    if abs(target - self.fluid.time) > 1e-9:
                        self.seek_recorded(target)
                        phase = ("complete" if target >= self.end_time - 1e-8
                                 else self.geometry.phase(target))
                        if phase != self.phase_name or target - self.history[-1]["time"] >= .25:
                            self.phase_name = phase
                            self.sample()
                else:
                    self.advance()
                if self.fluid.time >= self.end_time - 1e-8:
                    timeline.pause()
                    if not self._finished:
                        self._finished = True
                        result = self.save_report()
                        print(f"Robot pour finished: received "
                              f"{100 * result['final']['tube_transfer_fraction']:.1f}% of tube water.", flush=True)
            except Exception:
                timeline.pause()
                raise
            finally:
                self.busy = False

        def on_timeline(event):
            if not self._owns_scene() or getattr(self, "_exporting", False):
                return
            kind = int(event.type)
            if kind == int(omni.timeline.TimelineEventType.STOP):
                self.reset()
            elif kind == int(omni.timeline.TimelineEventType.PLAY) and self._finished:
                timeline.set_current_time(0.0)
                self.reset()

        omni.kit.app._robot_pour_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            on_update, name="robot test-tube pouring")
        omni.kit.app._robot_pour_timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            on_timeline, name="robot pour reset")

    def show_controls(self):
        import omni.kit.app
        import omni.timeline
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        previous = getattr(omni.kit.app, "_robot_pour_window", None)
        if previous is not None:
            previous.destroy()
        timeline = omni.timeline.get_timeline_interface()

        def view(name):
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.cameras[name]

        window = ui.Window("Robot pouring", width=310, height=240,
                           position_x=70, position_y=650)
        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Franka - test tube into beaker", height=24)
                self.status_label = ui.Label("Ready - press Play", height=40)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("Play", clicked_fn=timeline.play)
                    ui.Button("Pause", clicked_fn=timeline.pause)
                    ui.Button("Reset", clicked_fn=timeline.stop)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("Overview", clicked_fn=lambda: view("overview"))
                    ui.Button("Pour close-up", clicked_fn=lambda: view("close"))
                ui.Button(f"Export {VIDEO_SECONDS:g}s MP4", height=28, clicked_fn=self.export_video)
                ui.Label("Kinematic grip; " + ("recorded liquid simulation"
                         if LIQUID_MODE == "replay" else "live liquid simulation"), height=20)
        omni.kit.app._robot_pour_window = window
        self.sample()


async def create_demo(new_stage=True, attach=True):
    import carb
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    _enable_dependencies()
    looks = _load_file("robot_pour_existing_looks", INSTALL / "droplet_milkcrown_newton_surface.py")
    epoch = looks._begin_water_demo("robot_pour")
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_current_time(0.0)
    context = omni.usd.get_context()
    if new_stage or context.get_stage() is None:
        # A synchronous switch cannot commit later after another example starts.
        context.new_stage()
    stage = context.get_stage()
    stage_id = context.get_stage_id()

    def is_current():
        return looks._demo_is_current("robot_pour", epoch, stage_id)

    def require_current():
        if not is_current():
            raise asyncio.CancelledError("Robot build was superseded by another example.")

    require_current()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if stage.GetPrimAtPath(ROOT).IsValid():
        stage.RemovePrim(ROOT)
    UsdGeom.Xform.Define(stage, ROOT)
    geometry, robots, fluids = _load_parts()
    if not math.isfinite(MOTION_TIME_SCALE) or MOTION_TIME_SCALE <= 0:
        raise ValueError("MOTION_TIME_SCALE must be finite and positive.")
    geometry.DURATION *= MOTION_TIME_SCALE
    geometry.MOTION_KNOTS = tuple((t * MOTION_TIME_SCALE, p, angle)
                                for t, p, angle in geometry.MOTION_KNOTS)
    water, tube_op, tube_data, beaker_data, cameras = _build_visuals(stage, geometry, looks)
    print("Preparing the official Franka and checking its complete pouring trajectory...", flush=True)
    rig = await robots.build(stage, ROOT + "/Franka", ROBOT_BASE,
                             geometry.TUBE_RADIUS + geometry.TUBE_THICKNESS,
                             is_current=is_current)
    require_current()
    rig.bake(geometry.hand_pose, _demo_end_time(geometry), fps=30)
    # Playback writes cached Xforms; PhysX must not create static/static joints.
    for prim in Usd.PrimRange(rig.root):
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            # Preserve the imported joint metadata, but remove PhysX joint
            # types after IK. The cached visual playback owns these links.
            prim.SetTypeName("Scope")
    points, groups = geometry.initial_water_points(PARTICLE_SPACING)
    fluid = fluids.PourFluid(points, groups, tube_data, beaker_data, geometry.tube_pose,
        spacing=PARTICLE_SPACING, voxel=COLLISION_GRID_SIZE, dt=1.0 / SIMULATION_FPS,
        substeps=SUBSTEPS, surface_voxel=SURFACE_VOXEL_SIZE,
        surface_max_cells=SURFACE_MAX_GRID_CELLS)
    if LIQUID_MODE == "replay":
        fluid = RecordedFluid(fluid, CACHE_FILE, geometry)
    elif LIQUID_MODE != "simulate":
        raise ValueError("LIQUID_MODE must be 'replay' or 'simulate'.")
    demo = PourDemo(stage, geometry, rig, fluid, water, tube_op, cameras)
    require_current()
    omni.kit.app._robot_testtube_demo = demo
    if attach:
        demo.attach()
        if SHOW_CONTROL_PANEL:
            demo.show_controls()
    context.get_selection().clear_selected_prim_paths()
    carb.settings.get_settings().set("/app/transform/operation", "select")
    initial = demo.history[0]
    print(f"Robot pour ready: {initial['initial_tube_volume_ml']:.1f} mL in the tube, "
          f"{fluid.particle_count} particles. Play / Pause / Stop work as in the crown example.", flush=True)
    return demo


async def validate_demo(demo, capture=True):
    """Exercise the real pour, collect mass/transfer evidence, and check Reset."""
    import omni.kit.app
    import omni.timeline
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    app = omni.kit.app.get_app()
    viewport = get_active_viewport()
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = demo.fluid.positions()
    shots = []

    async def shot(name, camera="overview"):
        if not capture or viewport is None:
            return
        demo.sync_visuals(force_surface=True)
        viewport.camera_path = demo.cameras[camera]
        await viewport.wait_for_rendered_frames(2)
        for _ in range(90):
            await app.next_update_async()
        path = OUTPUT / (name + ".png")
        await capture_viewport_to_file(viewport, file_path=str(path)).wait_for_result()
        shots.append(str(path))

    await shot("ready")
    got_flow = False
    progress = OUTPUT / "metrics.json"
    for index in range(int(round(demo.end_time * SIMULATION_FPS))):
        demo.advance()
        await app.next_update_async()
        if index % 15 == 0:
            sample = demo.sample()
            progress.write_text(json.dumps({"complete": False, "current": sample}, indent=2), encoding="utf-8")
            if not got_flow and demo.fluid.time > 2.2 * MOTION_TIME_SCALE and sample["tube_airborne"] >= 12:
                await shot("pouring", "overview")
                await shot("pouring_close", "close")
                got_flow = True
    if not got_flow:
        await shot("pouring_close", "close")
    await shot("finished")
    result = demo.save_report()
    result["complete"] = True
    result["captures"] = shots
    demo.reset()
    result["reset_max_position_error_m"] = float(np.max(np.abs(demo.fluid.positions() - initial)))
    result["reset_passed"] = result["reset_max_position_error_m"] < 1e-7
    result["success"] = result["success"] and result["reset_passed"]
    progress.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if viewport is not None:
        viewport.camera_path = demo.cameras["overview"]
    return result


def launch():
    """Run from Isaac Sim Script Editor; build once, then use Play/Pause/Stop."""
    import omni.kit.app
    request_epoch = getattr(omni.kit.app, "_water_demo_epoch", 0)

    async def build_once():
        try:
            if getattr(omni.kit.app, "_water_demo_epoch", 0) != request_epoch:
                raise asyncio.CancelledError("A newer water example was run.")
            return await create_demo()
        except Exception as error:
            if getattr(omni.kit.app, "_robot_pour_build_task", None) is not asyncio.current_task():
                raise asyncio.CancelledError("A newer water example was run.") from None
            OUTPUT.mkdir(parents=True, exist_ok=True)
            import traceback
            (OUTPUT / "metrics.json").write_text(json.dumps({
                "success": False, "build_error": str(error), "traceback": traceback.format_exc()
            }, indent=2), encoding="utf-8")
            raise

    previous = getattr(omni.kit.app, "_robot_pour_build_task", None)
    if previous is not None and not previous.done():
        previous.cancel()
    task = asyncio.ensure_future(build_once())
    omni.kit.app._robot_pour_build_task = task
    return task


def _entrypoint():
    # A .py can be opened in Script Editor, or run with python.bat directly.
    # Standalone default is visible. Headless validation is explicit opt-in.
    try:
        import omni.kit.app
        in_kit = omni.kit.app.get_app().is_running()
    except Exception:
        in_kit = False
    if in_kit:
        launch()
        return
    from isaacsim import SimulationApp
    test = "--headless-test" in sys.argv
    simulation_app = SimulationApp({"headless": test, "width": 1280, "height": 720,
                                    "renderer": "RaytracedLighting"})
    task = asyncio.ensure_future(create_demo(attach=not test))
    while simulation_app.is_running() and not task.done():
        simulation_app.update()
    demo = task.result()
    if test:
        validation = asyncio.ensure_future(validate_demo(demo))
        while simulation_app.is_running() and not validation.done():
            simulation_app.update()
        result = validation.result()
        print(json.dumps({"success": result["success"], "final": result["final"]}, indent=2), flush=True)
        simulation_app.close()
        if not result["success"]:
            raise SystemExit(1)
    else:
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()


if __name__ == "__main__":
    _entrypoint()
