"""Official offline Franka with RobotPoser LM and cached kinematic playback.

The robot is a kinematic demonstrator, not a contact-driven grasp simulation.
No World, SimulationContext, SimulationManager, Lula, or physics tensor view is
created. Tube and fluid physics belong to the caller's Newton scene.

Usage while the timeline is stopped:
    robot = await build(stage, '/World/Robot', (0, -.40, 0), .014)
    metrics = robot.bake(hand_pose_fn, duration=8.0, fps=30)
Then call robot.update(t) alongside the Newton fluid step.

hand_pose_fn(t) returns WORLD hand position and quaternion WXYZ, either as
(position, quaternion) or {'position': position, 'orientation': quaternion}.
It targets panda_hand, not the grasp-site center. For a side grip:
    R_hand = R_tube @ Ry(pi/2)
    p_hand = p_grip - R_hand @ [0, 0, .1034]
"""

import json
import math
from pathlib import Path

from lab_paths import PROJECT_ROOT, ISAAC_SIM_ROOT

import numpy as np


FRANKA_PACKAGE = ISAAC_SIM_ROOT / 'exts' / 'isaacsim.asset.importer.urdf' / 'data' / 'urdf' / 'robots' / 'franka_description'
FRANKA_CACHE = PROJECT_ROOT / 'assets' / 'robot_testtube_pour' / 'franka'
GRASP_OFFSET = np.array([0.0, 0.0, 0.1034])


class RobotIKBakeError(RuntimeError):
    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def _normal_quat(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError('Expected a finite quaternion in WXYZ order.')
    length = float(np.linalg.norm(value))
    if length < 1e-10:
        raise ValueError('Quaternion has zero length.')
    return value / length


def _slerp(q0, q1, alpha):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > .9995:
        return _normal_quat((1.0-alpha)*q0 + alpha*q1)
    angle = math.acos(dot)
    return (math.sin((1.0-alpha)*angle)*q0 + math.sin(alpha*angle)*q1) / math.sin(angle)


def _matrix(position, quaternion):
    from pxr import Gf
    q = _normal_quat(quaternion)
    result = Gf.Matrix4d(1.0)
    result.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
    result.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
    return result


def _split_matrix(matrix):
    from pxr import Gf
    tf = Gf.Transform(matrix)
    q = tf.GetRotation().GetQuat()
    return np.asarray(tf.GetTranslation(), dtype=float), _normal_quat([q.GetReal(), *q.GetImaginary()])


def _read_hand_pose(value):
    if isinstance(value, dict):
        position = value['position']
        quaternion = value.get('orientation', value.get('quaternion_wxyz'))
    else:
        position, quaternion = value
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Expected a finite world hand position of shape (3,).')
    return position, _normal_quat(quaternion)


async def _offline_franka_asset():
    import omni.kit.app
    from pxr import Usd
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in ('isaacsim.robot.schema', 'isaacsim.asset.importer.urdf', 'isaacsim.robot.poser'):
        manager.set_extension_enabled_immediate(extension, True)
    await omni.kit.app.get_app().next_update_async()

    source = FRANKA_PACKAGE / 'robots' / 'panda_arm_hand.urdf'
    if not source.is_file():
        raise FileNotFoundError(f'The bundled official Franka URDF is missing: {source}')
    manifest_path = FRANKA_CACHE / 'asset_manifest.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            candidate = Path(manifest['asset_path'])
            cached = Usd.Stage.Open(str(candidate)) if candidate.is_file() else None
            if cached and cached.GetDefaultPrim():
                return str(candidate)
        except (OSError, ValueError, KeyError, RuntimeError):
            pass

    from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
    FRANKA_CACHE.mkdir(parents=True, exist_ok=True)
    cfg = URDFImporterConfig()
    cfg.urdf_path = str(source)
    cfg.usd_path = str(FRANKA_CACHE)
    cfg.ros_package_paths = [{'name': 'franka_description', 'path': str(FRANKA_PACKAGE)}]
    cfg.fix_base = True
    cfg.merge_fixed_joints = False
    cfg.run_asset_transformer = False
    cfg.run_multi_physics_conversion = False
    cfg.allow_self_collision = False
    asset = str(URDFImporter(cfg).import_urdf())
    converted = Usd.Stage.Open(asset)
    if not converted or not converted.GetDefaultPrim():
        raise RuntimeError(f'Franka URDF conversion did not produce a valid default prim: {asset}')
    manifest_path.write_text(json.dumps({'asset_path': asset, 'source_urdf': str(source),
        'mode': 'official_local_URDF', 'physics': 'kinematic_replay'}, indent=2), encoding='utf-8')
    return asset


def _restore_fixed_urdf_frames(stage, robot_prim):
    """Restore inertial-less fixed frames omitted by the 6.2 importer.

    Its official Panda import retains panda_link8 as an Xform but omits
    panda_joint8; panda_hand_joint still names link8, disconnecting the schema
    tree. Recreate only missing fixed URDF joints from the original zero-pose
    Xforms. This scene-local repair does not invent movable robot joints.
    """
    import xml.etree.ElementTree as ET
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from usd.schema.isaac.robot_schema import Classes, Relations
    by_name = {p.GetName(): p for p in Usd.PrimRange(robot_prim)}
    urdf = ET.parse(FRANKA_PACKAGE / 'robots' / 'panda_arm_hand.urdf').getroot()
    repairs = []
    for source_joint in urdf.findall('joint'):
        if source_joint.get('type') != 'fixed' or source_joint.get('name') in by_name:
            continue
        parent = by_name.get(source_joint.find('parent').get('link'))
        child = by_name.get(source_joint.find('child').get('link'))
        if parent is None or child is None:
            continue
        local = (UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0) *
                 UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0).GetInverse())
        position, quaternion = _split_matrix(local)
        for link in (parent, child):
            link.AddAppliedSchema(Classes.LINK_API.value)
            rel = robot_prim.GetRelationship(Relations.ROBOT_LINKS.name)
            targets = list(rel.GetTargets())
            if link.GetPath() not in targets:
                rel.SetTargets(targets + [link.GetPath()])
        joint = UsdPhysics.FixedJoint.Define(stage, robot_prim.GetPath().AppendPath('Physics/'+source_joint.get('name')))
        joint.GetPrim().AddAppliedSchema(Classes.JOINT_API.value)
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([child.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, position)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*map(float, quaternion[1:]))))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
        robot_prim.GetRelationship(Relations.ROBOT_JOINTS.name).AddTarget(joint.GetPath())
        repairs.append(source_joint.get('name'))
    return repairs


class KinematicFranka:
    def __init__(self, stage, root, robot_prim, hand, base, links, poser, finger_states, asset_path):
        self.stage, self.root, self.robot_prim = stage, root, robot_prim
        self.hand, self.base, self.links, self.poser = hand, base, links, poser
        self.finger_states, self.asset_path = finger_states, asset_path
        self.times = None
        self.samples = None
        self._ops = None
        self.metrics = None

    def bake(self, pose_fn, duration, fps=30):
        """Solve all frames while stopped. Failed IK raises with numeric metrics.

        Stores successful poses as IsaacNamedPose prims. Runtime only interpolates
        the saved rigid link transforms and never invokes articulation control.
        """
        import carb
        import omni.timeline
        from pxr import Gf, UsdGeom
        from isaacsim.robot.poser import Transform, store_named_pose
        if omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError('Stop or pause the timeline before baking robot poses.')
        if duration <= 0 or fps <= 0:
            raise ValueError('duration and fps must be positive.')
        count = int(math.ceil(float(duration)*float(fps))) + 1
        times = np.linspace(0.0, float(duration), count)
        snapshots = [Gf.Matrix4d(UsdGeom.Xformable(p).GetLocalTransformation()) for p in self.links]
        samples = np.empty((count, len(self.links), 7), dtype=float)
        previous = None
        max_position_error, max_orientation_error, max_joint_delta = 0.0, 0.0, 0.0
        failures, solved = [], 0
        import xml.etree.ElementTree as ET
        source_robot = ET.parse(FRANKA_PACKAGE / 'robots' / 'panda_arm_hand.urdf').getroot()
        velocity_limits = {j.get('name'): float(j.find('limit').get('velocity'))
                           for j in source_robot.findall('joint') if j.find('limit') is not None}
        arm_names = [j.prim_path.rsplit('/', 1)[-1] for j in self.poser.joints]
        peak_velocities = {name: 0.0 for name in arm_names}
        peak_intervals = {name: [0.0, 0.0] for name in arm_names}
        arm_velocity_limits = {name: velocity_limits[name] for name in arm_names}
        metrics = {'success': False, 'asset_path': self.asset_path, 'frames_requested': count,
            'fps': float(fps), 'duration_seconds': float(duration), 'mode': 'RobotPoser_LM_kinematic_replay',
            'robot_path': str(self.robot_prim.GetPath()), 'hand_path': str(self.hand.GetPath()),
            'grasp_site': str(self.hand.GetPath())+'/PourGraspSite',
            'restored_fixed_frames': list(getattr(self, 'restored_fixed_frames', [])), 'failures': failures,
            'peak_joint_velocities_rad_s': peak_velocities, 'urdf_joint_velocity_limits_rad_s': arm_velocity_limits,
            'peak_joint_velocity_intervals_s': peak_intervals,
            'velocity_limit_compliance_enforced': False}
        named_indices = set(np.linspace(0, count-1, min(9, count), dtype=int).tolist())
        try:
            for i, t in enumerate(times):
                world_position, world_quaternion = _read_hand_pose(pose_fn(float(t)))
                # RobotPoser targets are in robot-root coordinates, not world.
                robot_world = UsdGeom.Xformable(self.robot_prim).ComputeLocalToWorldTransform(0)
                local_matrix = _matrix(world_position, world_quaternion) * robot_world.GetInverse()
                local_position, local_quaternion = _split_matrix(local_matrix)
                target = Transform(t=local_position, q=local_quaternion)
                result = self.poser.solve_ik(target, seed=previous, tolerance=3e-4,
                    iters=100, null_space_bias=0.0)
                if not result.success:
                    # Explicitly retry the deterministic cold-start ladder.
                    self.poser.set_seed(None)
                    result = self.poser.solve_ik(target, tolerance=3e-4,
                        iters=180, null_space_bias=0.0)
                if not result.success:
                    failures.append({'frame': i, 'time': float(t), 'reason': 'LM IK did not converge'})
                    raise RobotIKBakeError(f'Franka IK failed at t={t:.3f}s; do not play an invalid grasp.', metrics)
                if previous:
                    dt = float(times[i]-times[i-1])
                    for key, value in result.joints.items():
                        name = key.rsplit('/', 1)[-1]
                        speed = abs(value-previous[key])/dt
                        if speed > peak_velocities[name]:
                            peak_velocities[name] = float(speed)
                            peak_intervals[name] = [float(times[i-1]), float(t)]
                    delta = max(abs(result.joints[k]-previous[k]) for k in result.joints)
                    max_joint_delta = max(max_joint_delta, delta)
                    if delta > 0.40:
                        failures.append({'frame': i, 'time': float(t), 'reason': 'IK branch discontinuity', 'max_joint_delta_rad': float(delta)})
                        raise RobotIKBakeError(f'Franka joint jump ({delta:.3f} rad) at t={t:.3f}s.', metrics)
                all_joints = dict(result.joints)
                all_joints.update(self.finger_states)
                self.poser.chain.teleport_anchored(all_joints)
                hand_world = UsdGeom.Xformable(self.hand).ComputeLocalToWorldTransform(0)
                actual_position, actual_quaternion = _split_matrix(hand_world)
                position_error = float(np.linalg.norm(actual_position-world_position))
                orientation_error = float(2*math.acos(np.clip(abs(np.dot(actual_quaternion,world_quaternion)),-1,1)))
                max_position_error = max(max_position_error, position_error)
                max_orientation_error = max(max_orientation_error, orientation_error)
                if position_error > .002 or orientation_error > .02:
                    failures.append({'frame': i, 'time': float(t), 'reason': 'USD FK readback outside tolerance',
                        'position_error_m': position_error, 'orientation_error_rad': orientation_error})
                    raise RobotIKBakeError(f'Franka FK mismatch at t={t:.3f}s: {position_error:.6f}m, {orientation_error:.6f}rad.', metrics)
                for j, link in enumerate(self.links):
                    p, q = _split_matrix(UsdGeom.Xformable(link).GetLocalTransformation())
                    samples[i,j,:3], samples[i,j,3:] = p, q
                if i in named_indices:
                    store_named_pose(self.stage, self.robot_prim, f'pour_frame_{i:04d}', result)
                previous = dict(result.joints)
                solved += 1
                if i % 30 == 0:
                    carb.log_info(f'RobotPoser pouring bake: {i+1}/{count} frames')
        except Exception:
            # Restore the original valid robot pose after any failed bake.
            for link, matrix in zip(self.links, snapshots):
                UsdGeom.Xformable(link).MakeMatrixXform().Set(matrix)
            metrics.update({'frames_solved': solved, 'max_hand_position_error_m': max_position_error,
                'max_hand_orientation_error_rad': max_orientation_error, 'max_joint_step_rad': max_joint_delta,
                'max_joint_velocity_rad_s': max(peak_velocities.values()),
                'max_joint_velocity_limit_ratio': max(peak_velocities[n]/arm_velocity_limits[n] for n in arm_names)})
            self.metrics = metrics
            raise
        self.times, self.samples = times, samples
        self._ops = [UsdGeom.Xformable(p).MakeMatrixXform() for p in self.links]
        metrics.update({'success': True, 'frames_solved': solved,
            'max_hand_position_error_m': max_position_error,
            'max_hand_orientation_error_rad': max_orientation_error,
            'max_joint_step_rad': max_joint_delta,
            'named_pose_count': len(named_indices), 'link_count': len(self.links),
            'max_joint_velocity_rad_s': max(peak_velocities.values()),
            'max_joint_velocity_limit_ratio': max(peak_velocities[n]/arm_velocity_limits[n] for n in arm_names)})
        self.metrics = metrics
        self.update(0.0)
        return metrics

    def update(self, t):
        """Replay rigid cached link Xforms; safe with caller-owned Newton stepping."""
        if self.times is None or self._ops is None:
            raise RuntimeError('Bake a valid robot trajectory before playback.')
        t = float(np.clip(t, 0.0, self.times[-1]))
        index = min(int(np.searchsorted(self.times, t, side='right'))-1, len(self.times)-2)
        index = max(index, 0)
        alpha = (t-self.times[index])/(self.times[index+1]-self.times[index])
        for j, op in enumerate(self._ops):
            a, b = self.samples[index,j], self.samples[index+1,j]
            position = (1-alpha)*a[:3] + alpha*b[:3]
            quaternion = _slerp(a[3:], b[3:], alpha)
            op.Set(_matrix(position, quaternion))


async def build(stage, robot_path='/World/Robot', base_position=(0.0,-.40,0.0), grip_radius=.014,
                *, is_current=None):
    """Build an official offline Franka for kinematic side-grip pouring.

    The caller must be stopped/paused. This helper affects only robot_path and
    the dedicated cached robot USD; it never replaces the active USD context.
    """
    import omni.kit.app
    import omni.timeline
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if omni.timeline.get_timeline_interface().is_playing():
        raise RuntimeError('Stop or pause the timeline before building the robot.')
    def require_current():
        if is_current is not None and not is_current():
            import asyncio
            raise asyncio.CancelledError('Robot scene changed during asset preparation.')

    require_current()
    asset_path = await _offline_franka_asset()
    require_current()
    from isaacsim.robot.poser import RobotPoser, validate_robot_schema
    from usd.schema.isaac.robot_schema import Classes
    if stage.GetPrimAtPath(robot_path).IsValid():
        raise ValueError(f'Robot destination already exists: {robot_path}')
    root = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
    root.GetReferences().AddReference(asset_path)
    UsdGeom.Xformable(root).MakeMatrixXform().Set(_matrix(base_position, [1,0,0,0]))
    await omni.kit.app.get_app().next_update_async()
    require_current()
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    candidates = [p for p in Usd.PrimRange(root) if validate_robot_schema(p)]
    if not candidates:
        raise RuntimeError('The converted Franka is missing IsaacRobotAPI.')
    robot_prim = min(candidates, key=lambda p: len(str(p.GetPath())))
    by_name = {}
    for prim in Usd.PrimRange(root):
        by_name.setdefault(prim.GetName(), []).append(prim)
    def find(name):
        candidates = by_name.get(name, [])
        if not candidates:
            raise RuntimeError(f'Official Franka link/joint not found: {name}')
        return min(candidates, key=lambda p: len(str(p.GetPath())))
    hand, base = find('panda_hand'), find('panda_link0')
    restored_frames = _restore_fixed_urdf_frames(stage, robot_prim)
    site = UsdGeom.Xform.Define(stage, hand.GetPath().AppendChild('PourGraspSite')).GetPrim()
    site.AddAppliedSchema(Classes.SITE_API.value)
    UsdGeom.Xformable(site).AddTranslateOp().Set(Gf.Vec3d(*GRASP_OFFSET))
    # This is deliberately kinematic replay. Disable dynamics only on this
    # robot so the main example's Newton-owned tube/fluid remain untouched.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        if "NewtonArticulationRootAPI" in prim.GetAppliedSchemas():
            prim.RemoveAppliedSchema("NewtonArticulationRootAPI")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    links = [p for p in Usd.PrimRange(robot_prim) if p.HasAPI(Classes.LINK_API.value) and p.IsA(UsdGeom.Xform)]
    if len(links) < 9:
        raise RuntimeError(f'Unexpected Franka link topology: {len(links)} schema links.')
    links.sort(key=lambda p: str(p.GetPath()))
    finger_states = {}
    for name in ('panda_finger_joint1','panda_finger_joint2'):
        finger_states[str(find(name).GetPath())] = float(np.clip(grip_radius, 0.0, .04))
    poser = RobotPoser(stage, robot_prim, base, hand)
    if len(poser.joints) != 7:
        raise RuntimeError(f'Expected 7 Franka arm joints, got {len(poser.joints)}.')
    result = KinematicFranka(stage,root,robot_prim,hand,base,links,poser,finger_states,asset_path)
    result.restored_fixed_frames = restored_frames
    return result
