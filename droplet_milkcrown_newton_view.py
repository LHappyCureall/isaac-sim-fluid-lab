# Milk crown with a left panel: Particles, Surface, or Both.
#
# Window > Script Editor > open this file from disk > Run.
# The liquid is the same Newton solver and the same knobs as
# droplet_milkcrown_newton.py. Particles are the spheres from
# droplet_milkcrown_newton_play.py. The surface is the water mesh from
# droplet_milkcrown_newton_surface.py. The buttons only show or hide those
# drawings. They do not restart the drop or change the solver.
#
# Press Play to start, Pause to freeze, Stop to return to the drop.
# The first surface build compiles and the window sits still.

import importlib.util

from lab_paths import PROJECT_ROOT

MODES = ("particles", "surface", "both")

_SURFACE_SPEC = importlib.util.spec_from_file_location(
    "droplet_milkcrown_newton_surface",
    str(PROJECT_ROOT / "droplet_milkcrown_newton_surface.py"),
)
surface = importlib.util.module_from_spec(_SURFACE_SPEC)
_SURFACE_SPEC.loader.exec_module(surface)
crown = surface.crown


def _shows_particles(mode):
    return mode in ("particles", "both")


def _shows_surface(mode):
    return mode in ("surface", "both")


def _run():
    import carb
    import omni.kit.app
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade, Vt

    previous_window = getattr(omni.kit.app, "_newton_view_window", None)
    if previous_window is not None:
        previous_window.destroy()
        omni.kit.app._newton_view_window = None

    epoch = surface._begin_water_demo("milkcrown_view")
    context = omni.usd.get_context()
    if context.get_stage() is None:
        context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Open a stage in Isaac Sim before running this script.")
    stage_id = context.get_stage_id()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    # Long on purpose. Each solver step takes seconds of wall clock, so a
    # 3 second timeline would hit Stop before the drop lands. The liquid
    # itself still ends at DURATION, then this script pauses.
    timeline.set_end_time(600.0)

    surface._setup_scene_lighting(stage)
    crown._add_wood_table(stage)
    surface._apply_table_material(stage)
    surface._configure_refraction()
    for path in ("/World/droplet", "/World/Cup", "/World/particleSystem"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

    args = crown.Example.create_parser().parse_args([])
    example = crown.Example(crown._KitViewer(), args)
    # The solver attaches collider_ids and other scratch arrays on the first step.
    # State.assign then refuses to copy back onto that grown state, so keep only
    # the particle fields that define the drop at rest.
    wp = crown.wp
    rest = {
        "particle_q": wp.clone(example.state_0.particle_q),
        "particle_qd": wp.clone(example.state_0.particle_qd),
        "particle_Jp": wp.clone(example.state_0.mpm.particle_Jp),
        "particle_qd_grad": wp.clone(example.state_0.mpm.particle_qd_grad),
        "particle_elastic_strain": wp.clone(example.state_0.mpm.particle_elastic_strain),
        "particle_stress": wp.clone(example.state_0.mpm.particle_stress),
    }

    root = "/World/NewtonCrown"
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(root)
    surface._add_glass_cup(stage, args.cup_radius, args.cup_height, crown.CUP_THICKNESS)

    spacing = example.particle_spacing
    particle_root = UsdGeom.Xform.Define(stage, f"{root}/Particles")
    color = Gf.Vec3f(*crown.LIQUID_COLOR)
    proto = UsdGeom.Sphere.Define(stage, f"{root}/Particles/Sphere")
    proto.CreateRadiusAttr(float(spacing * 0.5 * crown.SPHERE_SCALE))
    proto.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(
        crown._omnipbr(stage, "/World/Looks/NewtonLiquid", crown.LIQUID_COLOR, 1.0, 0.15, 0.6)
    )
    instancer = UsdGeom.PointInstancer.Define(stage, f"{root}/Particles/Liquid")
    instancer.CreatePrototypesRel().AddTarget(proto.GetPath())
    instancer.CreateProtoIndicesAttr(Vt.IntArray([0] * len(example.state_0.particle_q)))
    instancer.CreatePositionsAttr(Vt.Vec3fArray())

    surface_voxel = max(0.008, 0.75 * example.solver.voxel_size)
    surface_ctx = example.solver.create_particle_surface(
        voxel_size=surface_voxel,
        max_grid_cells=4_000_000,
        kernel_radius=max(3.0 * spacing, 1.5 * surface_voxel),
        threshold=0.22,
        smooth_lambda=0.25,
        anisotropic=True,
        kernel_scale=0.6,
        anisotropy_ratio=8.0,
        anisotropy_scale=1.6,
        anisotropy_min_neighbors=6,
        anisotropy_binning=True,
        anisotropy_strength=0.85,
        field_smooth_iterations=1,
        field_smooth_radius=1,
        mesh_smooth_iterations=1,
    )
    surface_mesh = UsdGeom.Mesh.Define(stage, f"{root}/Surface")
    surface_mesh.CreateDoubleSidedAttr(True)
    surface_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    surface_mesh.CreatePointsAttr(Vt.Vec3fArray())
    surface_mesh.CreateFaceVertexCountsAttr(Vt.IntArray())
    surface_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray())
    surface_mesh.CreateNormalsAttr(Vt.Vec3fArray())
    surface._apply_water_material(stage, surface_mesh)

    view = {"mode": "both", "surface_time": None}
    playing = {"on": False}
    busy = {"on": False}
    clock = {"t": timeline.get_current_time()}
    panel = {}

    def _visible(prim, shown):
        imageable = UsdGeom.Imageable(prim)
        if shown:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _apply_visibility():
        _visible(particle_root.GetPrim(), _shows_particles(view["mode"]))
        _visible(surface_mesh.GetPrim(), _shows_surface(view["mode"]))

    def _paint_buttons():
        buttons = panel.get("buttons")
        if not buttons:
            return
        for name, button in buttons.items():
            button.style = {
                "background_color": ui.color(0.12, 0.42, 0.30) if name == view["mode"] else ui.color(0.16, 0.16, 0.16)
            }

    def _set_status(text):
        label = panel.get("status")
        if label is None:
            return
        label.text = text

    def _positions():
        return Vt.Vec3fArray.FromNumpy(example.state_0.particle_q.numpy().astype("float32"))

    def _draw_particles():
        instancer.GetPositionsAttr().Set(_positions())

    def _draw_surface():
        if view["surface_time"] == example.sim_time:
            return
        extracted = example.solver.extract_particle_surface(example.state_0, surface_ctx)
        verts, indices, normals = extracted.to_arrays()
        if verts is None or indices is None:
            surface_mesh.GetPointsAttr().Set(Vt.Vec3fArray())
            surface_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray())
            surface_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray())
            view["surface_time"] = example.sim_time
            return
        points = verts.numpy().astype("float32")
        normal_values = normals.numpy().astype("float32") if normals is not None else None
        points, normal_values = surface._clip_water_floor(
            points, normal_values, args.cup_radius, args.cup_height
        )
        faces = indices.numpy().astype("int32")
        counts = crown.np.full(faces.shape[0] // 3, 3, dtype=crown.np.int32)
        surface_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        surface_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts))
        surface_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
        if normal_values is not None:
            surface_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normal_values))
            surface_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        view["surface_time"] = example.sim_time

    def _draw():
        if _shows_particles(view["mode"]):
            _draw_particles()
        if _shows_surface(view["mode"]):
            _draw_surface()

    def _set_mode(mode):
        if mode not in MODES:
            raise ValueError(f"View mode must be one of {MODES}.")
        if not surface._demo_is_current("milkcrown_view", epoch, stage_id):
            return
        view["mode"] = mode
        _apply_visibility()
        _paint_buttons()
        if not busy["on"]:
            _draw()
        _set_status(f"View: {mode}\nPlay starts the same liquid. This button only changes drawing.")
        print(f"Newton crown view: {mode}", flush=True)

    def _restore(state):
        state.particle_q.assign(rest["particle_q"])
        state.particle_qd.assign(rest["particle_qd"])
        state.mpm.particle_Jp.assign(rest["particle_Jp"])
        state.mpm.particle_qd_grad.assign(rest["particle_qd_grad"])
        state.mpm.particle_elastic_strain.assign(rest["particle_elastic_strain"])
        state.mpm.particle_stress.assign(rest["particle_stress"])

    def _reset():
        _restore(example.state_0)
        _restore(example.state_1)
        example.sim_time = 0.0
        view["surface_time"] = None
        clock["t"] = timeline.get_current_time()
        _draw()
        _set_status(f"View: {view['mode']}\nBack at the start.")
        print("Newton crown is back at the start.", flush=True)

    def _step_if_time_moved():
        if not surface._demo_is_current("milkcrown_view", epoch, stage_id):
            return
        if busy["on"]:
            return
        t = timeline.get_current_time()
        if t + 1.0e-8 < clock["t"]:
            _reset()
            return
        if t <= clock["t"] + 1.0e-8 or example.sim_time >= crown.DURATION:
            return
        clock["t"] = t
        busy["on"] = True
        try:
            example.step()
            _draw()
            if example.sim_time >= crown.DURATION and timeline.is_playing():
                playing["on"] = False
                timeline.pause()
                _set_status(f"View: {view['mode']}\nFinished at t={example.sim_time:.2f}s")
                print(f"Newton crown finished at t={example.sim_time:.3f}s", flush=True)
        finally:
            busy["on"] = False

    def _on_timeline(event):
        if not surface._demo_is_current("milkcrown_view", epoch, stage_id):
            return
        kind = int(event.type)
        if kind == int(omni.timeline.TimelineEventType.PLAY):
            if example.sim_time >= crown.DURATION:
                _reset()
            playing["on"] = True
            _set_status(f"View: {view['mode']}\nPlaying.")
            print("Play: Newton crown is stepping.", flush=True)
        elif kind == int(omni.timeline.TimelineEventType.PAUSE):
            playing["on"] = False
            _set_status(f"View: {view['mode']}\nPaused at t={example.sim_time:.2f}s")
            print(f"Pause at t={example.sim_time:.3f}s", flush=True)
        elif kind == int(omni.timeline.TimelineEventType.STOP):
            playing["on"] = False
            _reset()
        elif kind == int(omni.timeline.TimelineEventType.CURRENT_TIME_CHANGED):
            _step_if_time_moved()

    def _on_update(_event):
        _step_if_time_moved()

    crown._aim_camera(stage)
    carb.settings.get_settings().set("/app/transform/operation", "select")
    context.get_selection().clear_selected_prim_paths()

    window = ui.Window("Milk crown view", width=250, height=280, position_x=16, position_y=80)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Fluid drawing", height=22)
            panel["status"] = ui.Label("", word_wrap=True, height=48)
            panel["buttons"] = {}
            for mode, title in (("particles", "Particles"), ("surface", "Surface"), ("both", "Both")):
                panel["buttons"][mode] = ui.Button(
                    title, height=30, clicked_fn=lambda selected=mode: _set_mode(selected)
                )
            ui.Button("Play", height=28, clicked_fn=timeline.play)
            ui.Button("Pause", height=28, clicked_fn=timeline.pause)
            ui.Button("Stop", height=28, clicked_fn=timeline.stop)
    omni.kit.app._newton_view_window = window

    omni.kit.app._newton_play_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
        _on_timeline, name="newton crown view"
    )
    omni.kit.app._newton_play_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_update, name="newton crown view step"
    )
    print(
        "Building the water surface. The first extract compiles and the window will sit still.",
        flush=True,
    )
    _set_mode("both")
    print(
        "Milk crown view is waiting on the left. Particles, Surface, and Both only change drawing. "
        "Press Play to start, Pause to freeze, Stop to go back to the start. "
        f"It pauses itself at {crown.DURATION:.2f}s. "
        "Do not Run the other Newton crown scripts in this stage.",
        flush=True,
    )


if __name__ == "__main__":
    _run()
