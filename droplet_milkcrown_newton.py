# Milk-crown cup using the Isaac Sim bundled Newton implicit MPM solver.
#
# Same geometry as droplet_milkcrown.py: a drop falls onto a film of the same
# liquid inside a cup. Z is up, meters. PhysX PBD knobs (COHESION, VISCOSITY,
# SURFACE_TENSION) are not SI and are not copied. MPM viscosity is pascal-seconds.
#
# From the WSL env:
#   source ~/.venvs/newton/bin/activate
#   python ./droplet_milkcrown_newton.py --viewer gl
#   python ./droplet_milkcrown_newton.py --viewer null --num-frames 20
#   python ./droplet_milkcrown_newton.py --viewer usd \
#       --output-path ./droplet_out/newton/milkcrown.usd --num-frames 60

import os

from lab_paths import PROJECT_ROOT

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# --- knobs -----------------------------------------------------------------
# Lengths are meters, Z is up. Drop and film are the same liquid.
# In Isaac Sim: edit these, then close this Script Editor tab, reopen the
# file from disk, and Run. Run uses the open buffer, not the file on disk.
# These are MPM numbers. Do not copy COHESION or VISCOSITY from
# droplet_milkcrown.py. That viscosity of 80 is a PBD coefficient.
# Here VISCOSITY is pascal-seconds, and milk is about 0.0017.
RADIUS = 0.1              # drop radius, meters. Center height is DROP_HEIGHT
DROP_HEIGHT = 0.35        # fall of about 0.20 m. Impact speed is about 2 m/s
DROP_X = 0.0              # 0 keeps the drop on the axis of the cup
VOXEL_SIZE = 0.01        # MPM cell size, meters. This is the precision.
                          # Spacing is VOXEL_SIZE / PARTICLES_PER_CELL.
                          # 0.015 with 2 particles per cell is 7.5 mm.
                          # Halving VOXEL_SIZE is several times slower.
PARTICLES_PER_CELL = 2
SPHERE_SCALE = 1.1        # drawn ball radius is spacing * 0.5 * SPHERE_SCALE.
                          # This only changes how big each dot looks.
CUP_INNER_RADIUS = 0.4    # inner radius of the glass, meters
CUP_THICKNESS = 0.008     # drawn glass thickness. The collider is WALL_THICKNESS
CUP_HEIGHT = 0.48         # wall height. The rim is at this height
WALL_THICKNESS = 0.03     # collider thickness. Thicker than the drawn glass
FILM_DEPTH = 0.038        # film thickness, meters. Not a layer count
DENSITY = 1000.0          # kg/m^3. Milk is about 1000
VISCOSITY = 0.0017        # Pa·s. Milk is about 0.0017. Water is about 0.001.
                          # 50 is goo. Do not put the PhysX value 80 here.
TENSILE_YIELD_RATIO = 0.5 # how easily the sheet tears into droplets.
                          # Lower tears sooner. 0 is very brittle.
GLASS_OPACITY = 0.04      # 0 disappears, 0.04 is clear glass, higher turns milky
# RGB, each channel 0 to 1.
#   milk       (0.96, 0.96, 0.93)   fades into the glass
#   sky blue   (0.25, 0.55, 0.92)
#   deep blue  (0.12, 0.32, 0.72)
#   tea        (0.86, 0.52, 0.16)
#   juice      (0.95, 0.45, 0.12)
#   green      (0.20, 0.62, 0.38)
LIQUID_COLOR = (0.20, 0.62, 0.38)
VIEW = "side"             # side, front, high, close, or top
FPS = 60.0
SUBSTEPS = 4              # solver steps per frame. Higher is slower and stiffer
DURATION = 2.0            # seconds, then the Isaac run stops. The crown is in the first 1.5 s.
# ---------------------------------------------------------------------------

VIEWS = {
    "side": (28, (1.15, -1.55, 1.05), (0.0, 0.0, 0.16), (0.0, 0.0, 1.0)),
    "front": (28, (0.0, -2.10, 0.55), (0.0, 0.0, 0.20), (0.0, 0.0, 1.0)),
    "high": (26, (0.85, -1.15, 1.55), (0.0, 0.0, 0.12), (0.0, 0.0, 1.0)),
    "close": (35, (0.48, -0.62, 0.48), (0.0, 0.0, 0.16), (0.0, 0.0, 1.0)),
    "top": (22, (0.0, -0.02, 2.20), (0.0, 0.0, 0.05), (0.0, 1.0, 0.0)),
}


@wp.kernel
def _keep_in_cup(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    radius: float,
    floor_z: float,
):
    i = wp.tid()
    if (flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    pos = positions[i]
    vel = velocities[i]
    radial = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    if radial > radius:
        inv = 1.0 / radial
        direction = wp.vec3(pos[0] * inv, pos[1] * inv, 0.0)
        pos = wp.vec3(direction[0] * radius, direction[1] * radius, pos[2])
        outward = vel[0] * direction[0] + vel[1] * direction[1]
        if outward > 0.0:
            vel = wp.vec3(vel[0] - direction[0] * outward, vel[1] - direction[1] * outward, vel[2])
    if pos[2] < floor_z:
        pos = wp.vec3(pos[0], pos[1], floor_z)
        vel = wp.vec3(vel[0], vel[1], wp.max(vel[2], 0.0))
    positions[i] = pos
    velocities[i] = vel


def _disk_points(radius, z0, z1, spacing):
    xs = np.arange(-radius, radius + spacing * 0.5, spacing)
    ys = np.arange(-radius, radius + spacing * 0.5, spacing)
    zs = np.arange(z0 + 0.5 * spacing, z1, spacing)
    if zs.size == 0:
        zs = np.array([z0 + 0.5 * (z1 - z0)], dtype=np.float64)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = grid[:, 0] ** 2 + grid[:, 1] ** 2 <= radius**2
    return grid[keep]


def _ball_points(center, radius, spacing):
    n = int(np.ceil(2.0 * radius / spacing)) + 1
    axis = (np.arange(n) - 0.5 * (n - 1)) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.sum(grid**2, axis=1) <= radius**2
    return grid[keep] + np.asarray(center, dtype=np.float64)


def _cup_mesh(radius, height, thickness, segments=64):
    vertices = []
    triangles = []

    def add(x, y, z):
        vertices.append((float(x), float(y), float(z)))
        return len(vertices) - 1

    def quad(a, b, c, d):
        triangles.extend((a, b, c, a, c, d))

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    inner_lo, inner_hi, outer_lo, outer_hi = [], [], [], []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        inner_lo.append(add(radius * c, radius * s, 0.0))
        inner_hi.append(add(radius * c, radius * s, height))
        outer_lo.append(add((radius + thickness) * c, (radius + thickness) * s, -thickness))
        outer_hi.append(add((radius + thickness) * c, (radius + thickness) * s, height))

    for i in range(segments):
        j = (i + 1) % segments
        quad(inner_lo[j], inner_lo[i], inner_hi[i], inner_hi[j])
        quad(outer_lo[i], outer_lo[j], outer_hi[j], outer_hi[i])
        quad(inner_hi[i], inner_hi[j], outer_hi[j], outer_hi[i])
        quad(outer_lo[j], outer_lo[i], inner_lo[i], inner_lo[j])

    floor_center = add(0.0, 0.0, 0.0)
    for i in range(segments):
        j = (i + 1) % segments
        triangles.extend((floor_center, inner_lo[i], inner_lo[j]))

    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.cup_radius = args.cup_radius
        self.film_depth = args.film_depth

        spacing = args.voxel_size / args.particles_per_cell
        self.particle_spacing = spacing
        film_radius = args.cup_radius - 2.0 * spacing
        film = _disk_points(film_radius, 0.0, args.film_depth, spacing)
        drop = _ball_points((args.drop_x, 0.0, args.drop_height), args.drop_radius, spacing)
        points = np.concatenate([film, drop], axis=0)
        jitter = (np.random.default_rng(7).random(points.shape) - 0.5) * (0.15 * spacing)
        jitter[:, 2] = np.clip(points[:, 2] + jitter[:, 2], 0.5 * spacing, None) - points[:, 2]
        points = points + jitter
        print(
            f"film {len(film)}  drop {len(drop)}  spacing {spacing:.4f} m  voxel {args.voxel_size:.4f} m",
            flush=True,
        )

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        mass = float(spacing**3 * args.density)
        radius = 0.5 * spacing
        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[mass] * len(points),
            radius=[radius] * len(points),
        )

        vertices, indices = _cup_mesh(args.cup_radius, args.cup_height, args.wall_thickness)
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
            color=(0.75, 0.8, 0.82),
            opacity=GLASS_OPACITY,
        )
        builder.add_ground_plane(height=-args.wall_thickness, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)
        self.model.mpm.friction.fill_(0.0)
        self.model.mpm.viscosity.fill_(args.viscosity)
        self.model.mpm.tensile_yield_ratio.fill_(args.tensile_yield_ratio)

        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size
        config.max_iterations = args.max_iterations
        config.tolerance = args.tolerance
        config.strain_basis = "P0"
        config.collider_basis = "pic"
        config.velocity_basis = "Q1"
        self.solver = SolverImplicitMPM(self.model, config=config)
        self.solver.setup_collider(collider_projection_threshold=[0.25 * args.voxel_size])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.mpm.particle_Jp.fill_(1.0)
        self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.15, -1.7, 0.85), pitch=-18.0, yaw=145.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            wp.launch(
                _keep_in_cup,
                dim=self.state_1.particle_count,
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.model.particle_flags,
                    self.cup_radius - self.particle_spacing,
                    0.0,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        z = self.state_0.particle_q.numpy()[:, 2]
        print(f"t={self.sim_time:.3f}s  z=[{z.min():.3f},{z.max():.3f}]", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        z = self.state_0.particle_q.numpy()[:, 2]
        if z.min() < -2.0 * self.solver.voxel_size:
            raise ValueError(f"particles fell through the cup, z min {z.min():.4f}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=FPS)
        parser.add_argument("--substeps", type=int, default=SUBSTEPS)
        parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
        parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)
        parser.add_argument("--particles-per-cell", type=int, default=PARTICLES_PER_CELL)
        parser.add_argument("--max-iterations", type=int, default=40)
        parser.add_argument("--tolerance", type=float, default=1.0e-3)
        parser.add_argument("--density", type=float, default=DENSITY)
        parser.add_argument("--viscosity", type=float, default=VISCOSITY, help="MPM viscosity [Pa·s]. Milk is about 0.0017.")
        parser.add_argument("--tensile-yield-ratio", type=float, default=TENSILE_YIELD_RATIO)
        parser.add_argument("--cup-radius", type=float, default=CUP_INNER_RADIUS)
        parser.add_argument("--cup-height", type=float, default=CUP_HEIGHT)
        parser.add_argument("--wall-thickness", type=float, default=WALL_THICKNESS)
        parser.add_argument("--film-depth", type=float, default=FILM_DEPTH)
        parser.add_argument("--drop-radius", type=float, default=RADIUS)
        parser.add_argument("--drop-height", type=float, default=DROP_HEIGHT)
        parser.add_argument("--drop-x", type=float, default=DROP_X)
        return parser


class _KitViewer:
    """Stand-in for the Newton window. Isaac Sim already has a viewport."""

    show_particles = True

    def set_model(self, model):
        return None

    def set_camera(self, **kwargs):
        return None

    def begin_frame(self, time):
        return None

    def log_state(self, state):
        return None

    def end_frame(self):
        return None


def _omnipbr(stage, path, color, opacity, roughness, specular):
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdShade

    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_path=path,
    )
    shader = UsdShade.Shader.Get(stage, path + "/Shader")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(specular)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
    return UsdShade.Material.Get(stage, path)


WOOD_TEXTURE = str(PROJECT_ROOT / "textures" / "wooden_table.jpg")


def _add_wood_table(stage):
    """A wood plane under the cup. The Newton view has no ground, so it was black."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    for path in ("/World/ground", "/World/GroundPlane", "/World/groundPlane"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

    size = 4.0
    half = size * 0.5
    top = -CUP_THICKNESS - 0.004
    repeats = 2.5
    mesh = UsdGeom.Mesh.Define(stage, "/World/NewtonTable")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-half, -half, top),
                Gf.Vec3f(half, -half, top),
                Gf.Vec3f(half, half, top),
                Gf.Vec3f(-half, half, top),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * 4))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(repeats, 0.0),
                Gf.Vec2f(repeats, repeats),
                Gf.Vec2f(0.0, repeats),
            ]
        )
    )

    material = _omnipbr(stage, "/World/Looks/NewtonTable", (1.0, 1.0, 1.0), 1.0, 0.62, 0.2)
    shader = UsdShade.Shader.Get(stage, "/World/Looks/NewtonTable/Shader")
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    if os.path.isfile(WOOD_TEXTURE):
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(WOOD_TEXTURE))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _add_lights(stage):
    from pxr import Gf, UsdGeom, UsdLux

    def aim(light, rotate_xyz):
        xform = UsdGeom.Xformable(light.GetPrim())
        ops = xform.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3f(*rotate_xyz))
        else:
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))

    if not stage.GetPrimAtPath("/World/DistantLight").IsValid():
        sun = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        sun.CreateIntensityAttr().Set(2200)
        sun.CreateAngleAttr().Set(6.0)
        sun.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        aim(sun, (-48, 30, 0))
    if not stage.GetPrimAtPath("/World/FillLight").IsValid():
        fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
        fill.CreateIntensityAttr().Set(700)
        fill.CreateAngleAttr().Set(12.0)
        fill.CreateColorAttr().Set(Gf.Vec3f(0.82, 0.88, 0.95))
        aim(fill, (-25, -140, 0))
    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr().Set(450)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.78, 0.74, 0.68))


def _aim_camera(stage):
    from pxr import Gf, UsdGeom

    shot = VIEWS.get(VIEW)
    if shot is None:
        print(f"unknown VIEW {VIEW!r}, using side")
        shot = VIEWS["side"]
    focal, eye, target, up = shot
    cam = UsdGeom.Camera.Define(stage, "/World/NewtonCam")
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    cam.CreateFocalLengthAttr().Set(focal)
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = "/World/NewtonCam"
            import carb

            settings = carb.settings.get_settings()
            # The red and green lines through the origin are the viewport grid, not the liquid.
            for name in ("guide/grid", "guide/axis"):
                settings.set(f"/persistent/app/viewport/{viewport.id}/{name}/visible", False)
            settings.set("/app/viewport/grid/enabled", False)
    except Exception as exc:
        print(f"camera switch skipped: {exc}")


def _run_in_isaac():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Open a stage in Isaac Sim before running this script.")

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_current_time(0.0)
    timeline.set_end_time(DURATION)
    timeline.set_looping(False)
    _add_lights(stage)
    _add_wood_table(stage)
    # The PhysX droplet uses the same space. Leave it in the stage, but stop drawing it.
    for path in ("/World/droplet", "/World/Cup", "/World/particleSystem"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

    args = Example.create_parser().parse_args([])
    example = Example(_KitViewer(), args)
    points = example.state_0.particle_q.numpy()

    root = "/World/NewtonCrown"
    old = stage.GetPrimAtPath(root)
    if old.IsValid():
        stage.RemovePrim(root)

    cup_points, cup_faces = _cup_mesh(args.cup_radius, args.cup_height, CUP_THICKNESS)
    glass = UsdGeom.Mesh.Define(stage, f"{root}/Glass")
    glass.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cup_points))
    glass.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(cup_faces) // 3)))
    glass.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(cup_faces.astype(np.int32)))
    glass.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)]))
    glass.CreateDisplayOpacityAttr(Vt.FloatArray([GLASS_OPACITY]))
    glass.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(glass.GetPrim()).Bind(
        _omnipbr(stage, "/World/Looks/NewtonGlass", (0.0, 0.0, 0.0), GLASS_OPACITY, 0.0, 1.0)
    )
    UsdGeom.PrimvarsAPI(glass.GetPrim()).CreatePrimvar(
        "doNotCastShadows", Sdf.ValueTypeNames.Bool
    ).Set(True)

    color = Gf.Vec3f(*LIQUID_COLOR)
    proto = UsdGeom.Sphere.Define(stage, f"{root}/Sphere")
    proto.CreateRadiusAttr(float(example.particle_spacing * 0.5 * SPHERE_SCALE))
    proto.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(
        _omnipbr(stage, "/World/Looks/NewtonLiquid", LIQUID_COLOR, 1.0, 0.15, 0.6)
    )
    instancer = UsdGeom.PointInstancer.Define(stage, f"{root}/Liquid")
    instancer.CreatePrototypesRel().AddTarget(proto.GetPath())
    instancer.CreateProtoIndicesAttr(Vt.IntArray([0] * len(points)))
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))

    # The Kit App object is a C++ binding and rejects extra attributes.
    # Keep the subscription on the Python module so a second Run can replace it.
    previous = getattr(omni.kit.app, "_newton_crown_sub", None)
    if previous is not None:
        previous.unsubscribe()

    busy = {"on": False}
    stopped = {"on": False}

    def _stop():
        if stopped["on"]:
            return
        stopped["on"] = True
        sub = getattr(omni.kit.app, "_newton_crown_sub", None)
        omni.kit.app._newton_crown_sub = None
        if sub is not None:
            sub.unsubscribe()
        print(f"Newton crown stopped at t={example.sim_time:.3f}s", flush=True)

    def _on_update(_event):
        if stopped["on"] or example.sim_time >= DURATION:
            _stop()
            return
        if busy["on"]:
            return
        busy["on"] = True
        try:
            example.step()
            instancer.GetPositionsAttr().Set(
                Vt.Vec3fArray.FromNumpy(example.state_0.particle_q.numpy().astype(np.float32))
            )
            if example.sim_time >= DURATION:
                _stop()
        finally:
            busy["on"] = False

    _aim_camera(stage)
    omni.kit.app._newton_crown_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_update, name="newton crown"
    )
    print(
        "Newton crown is in /World/NewtonCrown. "
        f"radius={RADIUS} drop_height={DROP_HEIGHT} film={FILM_DEPTH} "
        f"viscosity={VISCOSITY} tensile_yield={TENSILE_YIELD_RATIO} "
        f"glass_opacity={GLASS_OPACITY} view={VIEW}. "
        f"Stops at {DURATION:.2f}s. Each frame takes a few seconds. Do not press Play.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        import omni.usd as _kit_usd
    except ImportError:
        _kit_usd = None
    if _kit_usd is not None and _kit_usd.get_context().get_stage() is not None:
        _run_in_isaac()
    else:
        viewer, args = newton.examples.init(Example.create_parser())
        newton.examples.run(Example(viewer, args), args)
