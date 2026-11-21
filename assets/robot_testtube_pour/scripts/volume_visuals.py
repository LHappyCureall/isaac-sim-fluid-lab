"""Display-only water contrast and equivalent-volume guides for dosing.

The blue style is a visibility aid, not a change to the simulated liquid.
Guide heights assume a cylindrical beaker at rest. They use the supplied
particle-volume measurements, never a rendered mesh volume or water height.
No function in this module moves particles, changes masses, or adds collisions.
"""

import math


TRAINING_WATER_COLOR = (0.025, 0.30, 0.72)
TARGET_COLOR = (1.0, 0.24, 0.025)
RECEIVED_COLOR = (0.015, 0.78, 0.92)


def _root_path(scene):
    return str(scene.water.GetPath()).rsplit("/", 1)[0]


def _positive_volume(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _make_material(stage, path, color, roughness=0.25, specular=0.5):
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Get(stage, path)
    if not material:
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR", mtl_path=path)
        material = UsdShade.Material.Get(stage, path)
    shader = UsdShade.Shader.Get(stage, path + "/Shader")
    if not material or not shader:
        raise RuntimeError(f"Could not create display material: {path}")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(specular)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    return material


def set_water_style(scene, style="training_blue"):
    """Bind visible glossy blue water, or restore the scene's clear material.

    ``scene`` supplies stage and water (a UsdGeom.Mesh). The return value is
    JSON-compatible display metadata. Glass container bindings are untouched.
    """
    from pxr import Gf, UsdShade, Vt

    if style not in ("training_blue", "clear"):
        raise ValueError("style must be 'training_blue' or 'clear'.")
    binding = UsdShade.MaterialBindingAPI.Apply(scene.water.GetPrim())
    if style == "training_blue":
        if not getattr(scene, "_training_water_original_material_path", None):
            original, _relationship = binding.ComputeBoundMaterial()
            if original:
                scene._training_water_original_material_path = str(original.GetPath())
        path = _root_path(scene) + "/Looks/TrainingBlueWater"
        material = _make_material(scene.stage, path, TRAINING_WATER_COLOR, 0.16, 0.65)
        color = TRAINING_WATER_COLOR
    else:
        path = getattr(scene, "_training_water_original_material_path", None)
        path = path or "/World/Looks/NewtonClearWater"
        material = UsdShade.Material.Get(scene.stage, path)
        if not material:
            raise RuntimeError("The scene's original clear-water material is unavailable.")
        color = (1.0, 1.0, 1.0)
    binding.Bind(material)
    scene.water.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    scene.water.CreateDisplayOpacityAttr(Vt.FloatArray([1.0]))
    return {"style": style, "material_path": str(material.GetPath()), "display_only": True}


class FlowParticles:
    """Show actual airborne source particles that a sparse SDF can omit.

    Spheres are display glyphs with one spacing as their diameter, not extra
    liquid or a volume sensor. Their centers are the current solver positions.
    No interpolated stream, collisions or physics attributes are created.
    """

    def __init__(self, scene, spacing):
        from pxr import Gf, UsdGeom, UsdShade, Vt

        self.scene, self.spacing = scene, float(spacing)
        path = _root_path(scene) + "/AirborneWaterDisplay"
        self.instances = UsdGeom.PointInstancer.Define(scene.stage, path)
        sphere = UsdGeom.Sphere.Define(scene.stage, path + "/Particle")
        sphere.CreateRadiusAttr(0.5 * self.spacing)
        material = _make_material(scene.stage, path + "/Looks/Blue", TRAINING_WATER_COLOR, 0.2)
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        self.instances.CreatePrototypesRel().SetTargets([sphere.GetPath()])
        self.instances.CreatePositionsAttr(Vt.Vec3fArray())
        self.instances.CreateProtoIndicesAttr(Vt.IntArray())
        self.instances.GetPrim().SetCustomDataByKey("displayOnly", True)
        self.instances.GetPrim().SetCustomDataByKey("source", "current_solver_airborne_tube_particle_centers")
        self.visible = True

    def update(self, positions, labels, source_time, visible=True):
        import numpy as np
        from pxr import UsdGeom, Vt

        geometry = self.scene.geometry
        points = np.asarray(positions, dtype=np.float64)
        delta = points - np.asarray(geometry.BEAKER_CENTER)
        beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= geometry.BEAKER_RADIUS ** 2)
                  & (delta[:, 2] >= 0.0) & (delta[:, 2] <= geometry.BEAKER_HEIGHT))
        origin, _ = geometry.tube_pose(source_time)
        tube = geometry._cavity_mask((points - origin) @ geometry.tube_rotation(source_time))
        mask = ((np.asarray(labels) == 1) & ~(beaker | tube)
                & (points[:, 2] > geometry.TABLE_TOP_Z + self.spacing))
        selected = points[mask].astype(np.float32)
        self.instances.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(selected))
        self.instances.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(selected)))
        self.instances.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
        self.visible = bool(visible)
        self.count = len(selected)
        return self.count


class VolumeGuide:
    """Two external short lines at true, unscaled equivalent fill heights.

    Orange is the target and cyan is the received amount. Left/right offsets
    keep both visible when the heights nearly coincide. The beaker starts with
    water, so each height includes ``initial_beaker_ml``. The actual measured
    quantity remains the supplied received_ml from the fixed-particle sensor.

    At the example's 65 mm radius, 20 mL changes height by only 1.51 mm; these
    guides support the numeric HUD and do not claim optical milliliter sensing.
    """

    legend = "Orange: target | Cyan: received | Equivalent fill levels"

    def __init__(self, scene, initial_beaker_ml):
        from pxr import Gf, UsdGeom, UsdShade, Vt

        self.scene = scene
        self.initial_beaker_ml = _positive_volume(initial_beaker_ml, "initial_beaker_ml")
        self.center = tuple(float(v) for v in scene.geometry.BEAKER_CENTER)
        self.radius = float(scene.geometry.BEAKER_RADIUS)
        self.height = float(scene.geometry.BEAKER_HEIGHT)
        if (len(self.center) != 3 or not all(math.isfinite(v) for v in self.center)
                or not math.isfinite(self.radius) or self.radius <= 0.0
                or not math.isfinite(self.height) or self.height <= 0.0):
            raise ValueError("The beaker geometry must be finite and positive.")
        self.area = math.pi * self.radius * self.radius
        self.ml_per_mm = self.area * 1000.0
        self.path = _root_path(scene) + "/VolumeGuide"
        root = UsdGeom.Xform.Define(scene.stage, self.path)
        root.GetPrim().SetCustomDataByKey("measurement", "display_only_equivalent_cylindrical_volume")
        root.GetPrim().SetCustomDataByKey("legend", self.legend)
        thickness = float(scene.geometry.BEAKER_THICKNESS)
        self.y = self.center[1] - self.radius - thickness - 0.002
        self.width = 0.0004
        self.lines = {}
        # No physics schemas are applied: these are only rendered line segments.
        for name, color in (("Target", TARGET_COLOR), ("Received", RECEIVED_COLOR)):
            line = UsdGeom.BasisCurves.Define(scene.stage, self.path + "/" + name)
            line.CreateTypeAttr(UsdGeom.Tokens.linear)
            line.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            line.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
            line.CreateWidthsAttr(Vt.FloatArray([self.width]))
            line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            line.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            material = _make_material(scene.stage, self.path + "/Looks/" + name, color, 0.35)
            UsdShade.MaterialBindingAPI.Apply(line.GetPrim()).Bind(material)
            self.lines[name] = line
        self.update(0.0, 0.0)

    def equivalent_height(self, added_ml):
        """Return world Z for a supplied added volume; no surface is sampled."""
        volume = self.initial_beaker_ml + _positive_volume(added_ml, "added_ml")
        return self.center[2] + volume * 1.0e-6 / self.area

    def _set_line(self, name, z):
        from pxr import Gf, UsdGeom, Vt

        # A real height, not a vertical magnification. The two colored bars use
        # different horizontal positions so close levels cannot hide each other.
        offsets = (0.014, 0.043) if name == "Target" else (-0.043, -0.014)
        x0, x1 = (self.center[0] + offset for offset in offsets)
        line = self.lines[name]
        line.CreatePointsAttr(Vt.Vec3fArray([
            Gf.Vec3f(x0, self.y, z), Gf.Vec3f(x1, self.y, z)]))
        half = 0.5 * self.width
        line.CreateExtentAttr(Vt.Vec3fArray([
            Gf.Vec3f(x0 - half, self.y - half, z - half),
            Gf.Vec3f(x1 + half, self.y + half, z + half)]))
        within_beaker = self.center[2] <= z <= self.center[2] + self.height
        line.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if within_beaker else UsdGeom.Tokens.invisible)

    def update(self, target_ml, received_ml):
        """Update display geometry only and return its explanatory metadata."""
        target_ml = _positive_volume(target_ml, "target_ml")
        received_ml = _positive_volume(received_ml, "received_ml")
        target_z = self.equivalent_height(target_ml)
        received_z = self.equivalent_height(received_ml)
        self._set_line("Target", target_z)
        self._set_line("Received", received_z)
        self.current = {
            "target_ml": target_ml, "received_ml": received_ml,
            "target_height_m": target_z, "received_height_m": received_z,
            "height_error_m": received_z - target_z, "ml_per_mm": self.ml_per_mm,
            "initial_beaker_ml": self.initial_beaker_ml,
            "display_only": True, "vertical_magnification": 1.0,
            "measurement_source": "supplied_particle_volume_not_rendered_surface",
        }
        return dict(self.current)
