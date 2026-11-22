"""Read-only kinematic bank export; no robot or stage pose writes."""
import json
from pathlib import Path
import sys
import numpy as np
from pxr import Gf, UsdGeom
import omni.kit.app
import omni.timeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
demo = getattr(omni.kit.app, "_robot_testtube_demo", None)
if demo is None or type(demo).__name__ != "TargetPourDemo" or getattr(demo, "policy", None) is not None:
    raise RuntimeError("Open this repository baseline target-volume demo and pause it before this read-only export.")
if (Path(demo.base.__file__).resolve() != ROOT / "robot_testtube_pour.py"
        or Path(demo.geometry.__file__).resolve() != ROOT / "assets/robot_testtube_pour/scripts/geometry.py"
        or not demo.scene._owns_scene()):
    raise RuntimeError("Refusing to export an unknown or stale scene.")
if not np.allclose([k[0] for k in demo.geometry.MOTION_KNOTS[:5]], [0., 1.4, 4.4, 6.4, 8.4], atol=1e-10):
    raise RuntimeError("Expected exactly one legacy source-time scale of 2.0.")
demo._require_current()
if demo.busy or demo._testing or omni.timeline.get_timeline_interface().is_playing():
    raise RuntimeError('Expected the owned paused idle demo.')
rig = demo.original_rig
paths = [str(p.GetPath()) for p in rig.links]
times, samples = np.asarray(rig.times), np.asarray(rig.samples)
if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0) or abs(times[0]) > 1e-10 or times[-1] < 8.4
        or samples.shape != (len(times), len(paths), 7) or not np.isfinite(samples).all()
        or not paths or any(not path.startswith("/World/RobotPourDemo/Franka/") for path in paths)):
    raise RuntimeError("The owned robot kinematic bank has an invalid clock, shape, path or finite state.")
indices = {path: i for i, path in enumerate(paths)}
stage = demo.stage


def local_matrix(sample):
    result = Gf.Matrix4d(1)
    result.SetRotate(Gf.Quatd(float(sample[3]), Gf.Vec3d(*map(float, sample[4:7]))))
    result.SetTranslateOnly(Gf.Vec3d(*map(float, sample[:3])))
    return result


def world(path, sample_index, cache):
    if path in cache:
        return cache[path]
    if path in ('', '/'):
        return Gf.Matrix4d(1)
    prim = stage.GetPrimAtPath(path)
    if path in indices:
        local = local_matrix(rig.samples[sample_index, indices[path]])
    else:
        xform = UsdGeom.Xformable(prim)
        local = xform.GetLocalTransformation(0) if xform else Gf.Matrix4d(1)
    answer = local * world(str(prim.GetParent().GetPath()), sample_index, cache)
    cache[path] = answer
    return answer


matrices = np.empty((len(rig.times), len(paths), 4, 4))
for i in range(len(rig.times)):
    cache = {}
    for j, path in enumerate(paths):
        matrices[i, j] = np.asarray(world(path, i, cache))
if not np.isfinite(matrices).all():
    raise RuntimeError("Nonfinite exported robot FK matrices.")
out = ROOT / "_cache/pour_training/fast_v3"
if (out / "legacy_kinematic_bank.npz").exists() or (out / "legacy_kinematic_bank.json").exists():
    raise FileExistsError("Preserve the existing robot bank; use a clean local preparation directory.")
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'legacy_kinematic_bank.npz', times=rig.times, samples=rig.samples,
    world_matrices_gf_row=matrices, link_paths=np.asarray(paths),
    current_world_matrices_gf_row=np.asarray([np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)) for p in rig.links]))
(out / 'legacy_kinematic_bank.json').write_text(json.dumps({'purpose': 'read_only_timing_feasibility',
    'times_are_scaled_legacy_geometry_source_seconds': True,
    'matrix_convention': 'Gf row vector; transpose for column convention',
    'metrics': rig.metrics, 'link_paths': paths}, indent=2), encoding='utf-8')
print('Read-only bank export complete:', matrices.shape)
