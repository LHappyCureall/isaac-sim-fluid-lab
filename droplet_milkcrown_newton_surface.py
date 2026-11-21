# Milk crown drawn as one water surface, not as spheres.
#
# Window > Script Editor > open this file > Run.
# Same Newton solver and the same knobs as droplet_milkcrown_newton.py.
# Press Play to start, Pause to freeze, Stop to return to the drop.
# Clear water is applied automatically; no separate material script is needed.

import importlib.util
from pathlib import Path

from lab_paths import PROJECT_ROOT

# Table options: "white_marble", "pale_oak", "gray_concrete",
# "white_matte", or "original_wood". Change this one line, reopen, and Run.
TABLE_STYLE = "white_matte"
TABLE_OPTIONS = {
    "white_marble": {"label": "White marble", "roughness": 0.30, "repeats": 2.0, "normal": 0.10},
    "pale_oak": {"label": "Pale oak", "roughness": 0.48, "repeats": 2.0, "normal": 0.15},
    "gray_concrete": {"label": "Gray concrete", "roughness": 0.75, "repeats": 2.0, "normal": 0.20},
    "white_matte": {"label": "Matte white", "roughness": 0.65, "repeats": 1.0, "normal": 0.0},
    "original_wood": {"label": "Original wood", "roughness": 0.62, "repeats": 2.5, "normal": 0.0},
}
TEXTURE_DIRECTORY = PROJECT_ROOT / "textures" / "table_options"
# Downloaded CC0 maps and source/license URLs: textures/table_options/sources.json.
# More bounces prevent dark truncated refraction paths in the water/cup.
REFRACTION_BOUNCES = 12
GLASS_IOR = 1.50
GLASS_ROUGHNESS = 0.015
# Keep the reconstructed liquid out of the solid glass floor and walls.
# This clips only the rendered boundary, without moving Newton particles.
WATER_FLOOR_CLEARANCE = 0.0005
FAST_REFRACTION_SHADOWS = True
# Broad white fill reduces the heavy dark shadow below transparent water.
SOFT_LIGHTING = True

# Clear water material. These settings only affect this surface script.
# Neutral transmission; try (0.98, 0.995, 1.0) for a subtle blue tint.
WATER_COLOR = (1.0, 1.0, 1.0)
WATER_IOR = 1.333
WATER_ROUGHNESS = 0.01
# OmniGlass "depth" is Volume Absorption Scale, not geometric thickness.
WATER_ABSORPTION_SCALE = 0.001
NEUTRALIZE_DOME_LIGHT = True

_SPEC = importlib.util.spec_from_file_location(
    "droplet_milkcrown_newton",
    str(PROJECT_ROOT / "droplet_milkcrown_newton.py"),
)
crown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crown)


def _demo_is_current(owner, epoch, stage_id=None):
    """Two standalone examples share one Kit timeline and must be exclusive."""
    import omni.kit.app
    import omni.usd

    if (getattr(omni.kit.app, "_water_demo_owner", None) != owner
            or getattr(omni.kit.app, "_water_demo_epoch", None) != epoch):
        return False
    return stage_id is None or omni.usd.get_context().get_stage_id() == stage_id


def _begin_water_demo(owner):
    """Stop the previous example before touching its prims or the timeline.

    Called only on Run, never when this file is imported for its materials.
    Generation invalidation also covers callbacks already queued by Kit.
    """
    import asyncio
    import omni.kit.app
    import omni.timeline
    import omni.usd

    app = omni.kit.app
    epoch = getattr(app, "_water_demo_epoch", 0) + 1
    app._water_demo_epoch, app._water_demo_owner = epoch, owner
    old_demo = getattr(app, "_robot_testtube_demo", None)
    if old_demo is not None:
        old_demo._closed, old_demo.busy = True, True
        export = getattr(old_demo, "export_task", None)
        if export is not None and not export.done():
            export.cancel()
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        current_task = None
    build = getattr(app, "_robot_pour_build_task", None)
    if build is not None and build is not current_task:
        if not build.done():
            build.cancel()
        app._robot_pour_build_task = None
    for name in ("_newton_crown_sub", "_newton_play_sub", "_newton_play_update_sub",
                 "_robot_pour_update_sub", "_robot_pour_timeline_sub"):
        previous = getattr(app, name, None)
        if previous is not None:
            previous.unsubscribe()
            setattr(app, name, None)
    window = getattr(app, "_robot_pour_window", None)
    if window is not None:
        window.destroy()
        app._robot_pour_window = None
    app._robot_testtube_demo = None
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_current_time(0.0)
    stage = omni.usd.get_context().get_stage()
    if stage is not None:
        for path in ("/World/RobotPourDemo", "/World/NewtonCrown", "/World/NewtonCam"):
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
    return epoch


def _apply_table_material(stage, style=None):
    """Switch tabletop appearance without restarting the Newton simulation."""
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    style = TABLE_STYLE if style is None else style
    if style not in TABLE_OPTIONS:
        raise ValueError(f"Unknown TABLE_STYLE {style!r}; choose from {tuple(TABLE_OPTIONS)}")
    preset = TABLE_OPTIONS[style]
    table = UsdGeom.Mesh.Get(stage, "/World/NewtonTable")
    if not table:
        crown._add_wood_table(stage)
        table = UsdGeom.Mesh.Get(stage, "/World/NewtonTable")
    table.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    color = (0.78, 0.78, 0.78) if style == "white_matte" else (1.0, 1.0, 1.0)
    path = f"/World/Looks/NewtonTable_{style}"
    material = UsdShade.Material.Get(stage, path)
    if not material:
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=path
        )
        material = UsdShade.Material.Get(stage, path)
    shader = UsdShade.Shader.Get(stage, path + "/Shader")
    if not material or not shader:
        raise RuntimeError(f"Could not create tabletop material {path}")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(preset["roughness"])
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    if style == "original_wood":
        albedo = Path(crown.WOOD_TEXTURE)
    elif style == "white_matte":
        albedo = None
    else:
        albedo = TEXTURE_DIRECTORY / f"{style}_albedo_2k.jpg"
    if albedo is not None:
        if not albedo.is_file():
            raise FileNotFoundError(f"Table texture is missing: {albedo}")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(albedo)))
    if style not in ("original_wood", "white_matte"):
        roughness = TEXTURE_DIRECTORY / f"{style}_roughness_2k.jpg"
        normal = TEXTURE_DIRECTORY / f"{style}_normal_gl_2k.jpg"
        if roughness.is_file():
            shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(roughness))
            )
            shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(0.65)
        if normal.is_file():
            shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(normal)))
            shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(preset["normal"])
            shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(False)
    repeats = preset["repeats"]
    UsdGeom.PrimvarsAPI(table.GetPrim()).GetPrimvar("st").Set(
        Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(repeats, 0),
                      Gf.Vec2f(repeats, repeats), Gf.Vec2f(0, repeats)])
    )
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(material)
    print(f"Table: {style} ({preset['label']})", flush=True)


def _setup_scene_lighting(stage):
    from pxr import Gf, UsdLux

    crown._add_lights(stage)
    if SOFT_LIGHTING:
        dome = UsdLux.DomeLight.Get(stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(1000.0)
        dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdLux.DistantLight.Get(stage, "/World/DistantLight").GetIntensityAttr().Set(600.0)
        UsdLux.DistantLight.Get(stage, "/World/FillLight").GetIntensityAttr().Set(200.0)


def _configure_refraction():
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx-transient/sceneDb/fastRefractionShadow/enabled", FAST_REFRACTION_SHADOWS)
    settings.set_int("/rtx/pathtracing/maxBounces", REFRACTION_BOUNCES)
    settings.set_int("/rtx/pathtracing/maxSpecularAndTransmissionBounces", REFRACTION_BOUNCES)
    settings.set_int("/rtx/rtpt/maxBounces", min(REFRACTION_BOUNCES, 6))
    settings.set_int("/rtx/rtpt/maxSpecularAndTransmissionBounces", min(REFRACTION_BOUNCES, 6))


def _closed_cup_mesh(radius, height, thickness, segments=96):
    """Return an outward-oriented, closed visual glass cup, in meters.

    The interior floor is z=0, the rim is z=height, and the outer floor
    is z=-thickness. This mesh is for rendering; keep solver colliders
    independent. Face indices are a flat int32 triangle array.
    """
    import numpy as np

    if radius <= 0.0 or height <= 0.0 or thickness <= 0.0:
        raise ValueError("Cup radius, height, and thickness must be positive.")
    if int(segments) != segments or segments < 3:
        raise ValueError("Cup segments must be an integer of at least 3.")
    segments = int(segments)
    vertices = []
    triangles = []

    def ring(r, z):
        indices = []
        for angle in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False):
            indices.append(len(vertices))
            vertices.append((float(r * np.cos(angle)), float(r * np.sin(angle)), float(z)))
        return indices

    def quad(a, b, c, d):
        triangles.extend((a, b, c, a, c, d))

    inner_floor = ring(radius, 0.0)
    inner_rim = ring(radius, height)
    outer_floor = ring(radius + thickness, -thickness)
    outer_rim = ring(radius + thickness, height)
    inner_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    outer_center = len(vertices)
    vertices.append((0.0, 0.0, -float(thickness)))

    for i in range(segments):
        j = (i + 1) % segments
        # Normals face the liquid on the inside and the air outside.
        quad(inner_floor[j], inner_floor[i], inner_rim[i], inner_rim[j])
        quad(outer_floor[i], outer_floor[j], outer_rim[j], outer_rim[i])
        # Upward-facing rim joins the two walls.
        quad(inner_rim[i], outer_rim[i], outer_rim[j], inner_rim[j])
        triangles.extend((inner_center, inner_floor[i], inner_floor[j]))
        triangles.extend((outer_center, outer_floor[j], outer_floor[i]))

    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)

def _apply_glass_material(stage, glass):
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdShade, Vt

    path = "/World/Looks/NewtonClearGlass"
    material = UsdShade.Material.Get(stage, path)
    if not material:
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim", mtl_url="OmniGlass.mdl", mtl_name="OmniGlass", mtl_path=path
        )
        material = UsdShade.Material.Get(stage, path)
    shader = UsdShade.Shader.Get(stage, path + "/Shader")
    if not material or not shader:
        raise RuntimeError("Could not create the clear glass material.")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(GLASS_IOR)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(GLASS_ROUGHNESS)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.001)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("cutout_opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    UsdShade.MaterialBindingAPI.Apply(glass.GetPrim()).Bind(material)
    glass.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)]))
    glass.CreateDisplayOpacityAttr(Vt.FloatArray([1.0]))


def _apply_water_material(stage, surface):
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

    MATERIAL_PATH = "/World/Looks/NewtonClearWater"

    material = UsdShade.Material.Get(stage, MATERIAL_PATH)
    if not material:
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_path=MATERIAL_PATH,
        )
        material = UsdShade.Material.Get(stage, MATERIAL_PATH)
    shader = UsdShade.Shader.Get(stage, MATERIAL_PATH + "/Shader")
    if not material or not shader:
        raise RuntimeError("Could not create the OmniGlass water material.")

    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*WATER_COLOR)
    )
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(WATER_IOR)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(
        WATER_ROUGHNESS
    )
    # The extracted mesh encloses liquid volume, including the round drop.
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(
        WATER_ABSORPTION_SCALE
    )
    shader.CreateInput("reflection_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 1.0, 1.0)
    )
    # Transparency comes from transmission and refraction, not alpha cutout.
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("cutout_opacity", Sdf.ValueTypeNames.Float).Set(1.0)

    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    surface.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*WATER_COLOR)]))
    surface.CreateDisplayOpacityAttr(Vt.FloatArray([1.0]))
    UsdGeom.PrimvarsAPI(surface.GetPrim()).CreatePrimvar(
        "enableFastRefractionShadow", Sdf.ValueTypeNames.Bool
    ).Set(FAST_REFRACTION_SHADOWS)

    if NEUTRALIZE_DOME_LIGHT:
        dome = UsdLux.DomeLight.Get(stage, "/World/DomeLight")
        if dome:
            dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    print(
        f"Clear water applied: OmniGlass, IOR={WATER_IOR}, roughness={WATER_ROUGHNESS}. "
        "For a still image, pause and choose RTX Interactive (Path Tracing).",
        flush=True,
    )


def _add_glass_cup(stage, radius, height, thickness):
    from pxr import Sdf, UsdGeom, Vt

    points, faces = _closed_cup_mesh(radius, height, thickness)
    glass = UsdGeom.Mesh.Define(stage, "/World/NewtonCrown/Glass")
    glass.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    glass.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(faces) // 3)))
    glass.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    glass.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    glass.CreateDoubleSidedAttr(False)
    # Smooth cylindrical walls while retaining sharp rim/floor creases.
    face_normals = []
    for index, face in enumerate(faces.reshape(-1, 3)):
        kind = index % 8
        if kind < 4:
            sign = -1.0 if kind < 2 else 1.0
            for vertex in points[face]:
                radial = crown.np.linalg.norm(vertex[:2])
                face_normals.append((sign * vertex[0] / radial, sign * vertex[1] / radial, 0.0))
        else:
            normal = (0.0, 0.0, -1.0 if kind == 7 else 1.0)
            face_normals.extend([normal] * 3)
    glass.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(crown.np.asarray(face_normals, dtype="float32")))
    glass.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdGeom.PrimvarsAPI(glass.GetPrim()).CreatePrimvar(
        "enableFastRefractionShadow", Sdf.ValueTypeNames.Bool
    ).Set(FAST_REFRACTION_SHADOWS)
    # Clear the old alpha-glass approximation's shadow override.
    shadow = glass.GetPrim().GetAttribute("primvars:doNotCastShadows")
    if shadow:
        shadow.Set(False)
    _apply_glass_material(stage, glass)
    return glass


def _clip_water_floor(points, normals=None, cup_radius=None, cup_height=None):
    below = points[:, 2] < WATER_FLOOR_CLEARANCE
    points[below, 2] = WATER_FLOOR_CLEARANCE
    if normals is not None:
        normals[below] = (0.0, 0.0, -1.0)
    radius = (crown.CUP_INNER_RADIUS if cup_radius is None else cup_radius) - WATER_FLOOR_CLEARANCE
    height = crown.CUP_HEIGHT if cup_height is None else cup_height
    radial = crown.np.linalg.norm(points[:, :2], axis=1)
    wall = (radial > radius) & (points[:, 2] < height)
    points[wall, :2] *= (radius / radial[wall])[:, None]
    if normals is not None:
        normals[wall, :2] = points[wall, :2] / radius
        normals[wall, 2] = 0.0
        normals[wall & below, 2] = -1.0
        lengths = crown.np.linalg.norm(normals[wall], axis=1)
        normals[wall] /= lengths[:, None]
    return points, normals


def apply_appearance(style=None):
    """Apply a table/cup look to the current scene without resetting the drop."""
    import omni.usd
    from pxr import UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Open a stage first.")
    surface = UsdGeom.Mesh.Get(stage, "/World/NewtonCrown/Surface")
    if not surface:
        raise RuntimeError("Run this surface script once before switching table looks.")
    _setup_scene_lighting(stage)
    _apply_table_material(stage, style)
    _add_glass_cup(stage, crown.CUP_INNER_RADIUS, crown.CUP_HEIGHT, crown.CUP_THICKNESS)
    surface.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    points = crown.np.array(surface.GetPointsAttr().Get(), dtype="float32")
    normals = crown.np.array(surface.GetNormalsAttr().Get(), dtype="float32")
    if len(points):
        points, normals = _clip_water_floor(points, normals if len(normals) == len(points) else None)
        surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        if normals is not None:
            surface.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    _apply_water_material(stage, surface)
    _configure_refraction()


def _run():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    epoch = _begin_water_demo("milkcrown")
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

    _setup_scene_lighting(stage)
    crown._add_wood_table(stage)
    _apply_table_material(stage)
    _configure_refraction()
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

    _add_glass_cup(stage, args.cup_radius, args.cup_height, crown.CUP_THICKNESS)

    color = Gf.Vec3f(*WATER_COLOR)
    spacing = example.particle_spacing
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
    surface_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
    _apply_water_material(stage, surface_mesh)

    playing = {"on": False}
    busy = {"on": False}
    # Movie Capture does not leave the timeline in Play. It calls set_current_time
    # once per saved frame. Step once for each forward move, including that scrub.
    clock = {"t": timeline.get_current_time()}

    def _show():
        extracted = example.solver.extract_particle_surface(example.state_0, surface_ctx)
        verts, indices, normals = extracted.to_arrays()
        if verts is None or indices is None:
            surface_mesh.GetPointsAttr().Set(Vt.Vec3fArray())
            surface_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray())
            surface_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray())
            return
        points = verts.numpy().astype("float32")
        normal_values = normals.numpy().astype("float32") if normals is not None else None
        points, normal_values = _clip_water_floor(points, normal_values, args.cup_radius, args.cup_height)
        faces = indices.numpy().astype("int32")
        counts = crown.np.full(faces.shape[0] // 3, 3, dtype=crown.np.int32)
        surface_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        surface_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts))
        surface_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
        if normal_values is not None:
            surface_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normal_values))
            surface_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

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
        clock["t"] = timeline.get_current_time()
        _show()
        print("Newton crown is back at the start.", flush=True)

    def _step_if_time_moved():
        if not _demo_is_current("milkcrown", epoch, stage_id):
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
            _show()
            if example.sim_time >= crown.DURATION and timeline.is_playing():
                playing["on"] = False
                timeline.pause()
                print(f"Newton crown finished at t={example.sim_time:.3f}s", flush=True)
        finally:
            busy["on"] = False

    def _on_timeline(event):
        if not _demo_is_current("milkcrown", epoch, stage_id):
            return
        kind = int(event.type)
        if kind == int(omni.timeline.TimelineEventType.PLAY):
            if example.sim_time >= crown.DURATION:
                _reset()
            playing["on"] = True
            print("Play: Newton crown is stepping.", flush=True)
        elif kind == int(omni.timeline.TimelineEventType.PAUSE):
            playing["on"] = False
            print(f"Pause at t={example.sim_time:.3f}s", flush=True)
        elif kind == int(omni.timeline.TimelineEventType.STOP):
            playing["on"] = False
            _reset()
        elif kind == int(omni.timeline.TimelineEventType.CURRENT_TIME_CHANGED):
            _step_if_time_moved()

    def _on_update(_event):
        _step_if_time_moved()

    crown._aim_camera(stage)
    # The red/green/blue arrows are the Move tool, drawn because the surface is selected.
    import carb

    carb.settings.get_settings().set("/app/transform/operation", "select")
    omni.usd.get_context().get_selection().clear_selected_prim_paths()
    omni.kit.app._newton_play_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
        _on_timeline, name="newton crown play"
    )
    omni.kit.app._newton_play_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_update, name="newton crown play step"
    )
    print("Building the water surface. The first extract compiles and the window will sit still.", flush=True)
    _show()
    assert not stage.GetPrimAtPath("/World/RobotPourDemo").IsValid()
    assert getattr(omni.kit.app, "_robot_pour_update_sub", None) is None
    assert getattr(omni.kit.app, "_robot_pour_timeline_sub", None) is None
    print("Milkcrown is active; robot pouring callbacks, controls and geometry are cleared.", flush=True)
    print(
        "Water surface is in /World/NewtonCrown/Surface. "
        "Press Play to start, Pause to freeze, Stop to go back to the start. "
        f"It pauses itself at {crown.DURATION:.2f}s.",
        flush=True,
    )


if __name__ == "__main__":
    _run()
