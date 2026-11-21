"""Bake a validated 60 Hz physical liquid cache for the pouring example.

Run with the Isaac Sim Python launcher, for example::

    python.bat tools/bake_pour_cache.py --headless

It also runs inside Kit with ``--exec``. Existing Kit tools can import this
module and ``await bake()``. The demo is built without playback callbacks or
controls. The previous cache is replaced only after simulation, validation,
reset, and a successful temporary-file round trip.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


_PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIRECTORY))


def _project_root():
    from lab_paths import PROJECT_ROOT

    return Path(PROJECT_ROOT)


def _load_base():
    path = _project_root() / "robot_testtube_pour.py"
    name = "fluid_lab_cache_bake_source"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the project's pouring example.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _geometry_metadata(geometry):
    # RecordedFluid compares this mapping exactly. Additional initialization
    # and motion parameters belong in separate metadata mappings.
    return {
        "tube_radius": float(geometry.TUBE_RADIUS),
        "tube_length": float(geometry.TUBE_LENGTH),
        "tube_thickness": float(geometry.TUBE_THICKNESS),
        "beaker_radius": float(geometry.BEAKER_RADIUS),
        "beaker_height": float(geometry.BEAKER_HEIGHT),
        "beaker_center": [float(x) for x in geometry.BEAKER_CENTER],
        "beaker_thickness": float(geometry.BEAKER_THICKNESS),
    }


def _atomic_save(path, positions, times, labels, metadata, *, require_current=None):
    """Publish the compressed cache only after checking its serialized form."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".pour_cache-", suffix=".npz", dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream, positions=positions, times=times, labels=labels,
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            saved = archive["positions"]
            valid = (
                set(archive.files) == {"positions", "times", "labels", "metadata"}
                and saved.dtype == np.float32 and saved.shape == positions.shape
                and np.isfinite(saved).all()
                and np.array_equal(saved, positions)
                and archive["times"].dtype == np.float64
                and np.array_equal(archive["times"], times)
                and archive["labels"].dtype == np.int32
                and np.array_equal(archive["labels"], labels)
                and json.loads(str(archive["metadata"].item())) == metadata
            )
            if not valid:
                raise RuntimeError("The temporary liquid cache failed verification.")
        if require_current is not None:
            require_current()
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _quaternion_error(first, second):
    import numpy as np

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(2 * math.acos(float(np.clip(abs(np.dot(first, second)), 0, 1))))


def _require_owned_scene(demo):
    if not demo._owns_scene():
        raise asyncio.CancelledError("Another example replaced the cache-baking scene.")


async def bake(cache_path=None, *, base=None):
    """Simulate the entire physical motion, validate it, and write a cache.

    Requires a running Isaac Sim Kit application. ``base`` may be an already
    imported project ``robot_testtube_pour`` module. No recorded fluid is used.
    The scene is reset after validation; no timeline callbacks are attached.
    Returns a JSON-compatible summary without machine-specific file paths.
    """
    import numpy as np
    import omni.kit.app
    import omni.timeline
    from pxr import UsdGeom

    root = _project_root()
    output = (root / "assets" / "robot_testtube_pour" / "pour_cache.npz"
              if cache_path is None else Path(cache_path))
    base = _load_base() if base is None else base
    base.LIQUID_MODE = "simulate"
    base.STOP_AFTER_UPRIGHT = False
    base.SHOW_CONTROL_PANEL = False
    base.SIMULATION_FPS = 60.0
    app = omni.kit.app.get_app()
    omni.timeline.get_timeline_interface().pause()
    started = time.perf_counter()
    demo = await base.create_demo(attach=False)
    _require_owned_scene(demo)
    fluid, geometry = demo.fluid, demo.geometry
    if isinstance(fluid, base.RecordedFluid):
        raise RuntimeError("Cache baking requires the live particle solver.")
    duration = float(geometry.DURATION)
    dt = float(fluid.dt)
    steps = int(round(duration / dt))
    if (not math.isfinite(duration) or duration <= 0
            or abs(dt - 1 / 60) > 1e-12
            or abs(steps * dt - duration) > 1e-9
            or abs(demo.end_time - duration) > 1e-9):
        raise RuntimeError("The bake must cover the complete motion at exactly 60 Hz.")
    labels = np.asarray(fluid.labels, dtype=np.int32).copy()
    initial = np.asarray(fluid.positions(), dtype=np.float32).copy()
    count = len(initial)
    spacing = float(base.PARTICLE_SPACING)
    expected_mass = count * 1000.0 * spacing ** 3
    frames = np.empty((steps + 1, count, 3), dtype=np.float32)
    times = np.empty(steps + 1, dtype=np.float64)
    frames[0], times[0] = initial, 0.0
    initial_tube = labels == 1
    tube_count = int(np.count_nonzero(initial_tube))
    if count == 0 or labels.shape != (count,) or tube_count == 0:
        raise RuntimeError("The fixed initial particle set is invalid.")
    origin, _ = geometry.tube_pose(0.0)
    initial_local = (initial[initial_tube] - origin) @ geometry.tube_rotation(0.0)
    initial_height = float(np.diff(np.quantile(initial_local[:, 2], [0.05, 0.95]))[0])
    if initial_height <= 0:
        raise RuntimeError("The initial water column has no measurable height.")
    min_height, min_retention = math.inf, math.inf
    hold_samples = 0
    max_grasp_position, max_grasp_rotation = 0.0, 0.0
    max_collider_position, max_collider_rotation = 0.0, 0.0
    max_below_table, max_mass_error = 0, 0.0
    initial_model_masses = np.asarray(fluid.model.particle_mass.numpy(), dtype=float)
    model_mass = float(initial_model_masses.sum())
    if (initial_model_masses.shape != (count,)
            or not np.isfinite(initial_model_masses).all()
            or np.any(initial_model_masses <= 0)
            or abs(model_mass - expected_mass) > max(1e-9, expected_mass * 1e-6)):
        raise RuntimeError("Initial particle masses do not match the liquid volume.")
    # Keep robot/collider Xforms current, but avoid expensive surface extraction
    # while collecting particles. The constructor and reset show the surface.
    old_surface_interval = base.SURFACE_EVERY_N_STEPS
    base.SURFACE_EVERY_N_STEPS = steps + 1
    try:
        for index in range(1, steps + 1):
            _require_owned_scene(demo)
            if not demo.advance():
                raise RuntimeError("The physical pour ended before the cache was complete.")
            _require_owned_scene(demo)
            current_time = float(fluid.time)
            points = np.asarray(fluid.positions(), dtype=np.float32)
            masses = np.asarray(fluid.model.particle_mass.numpy(), dtype=float)
            if (points.shape != initial.shape or not np.isfinite(points).all()
                    or fluid.particle_count != count
                    or not np.array_equal(fluid.labels, labels)
                    or not np.array_equal(masses, initial_model_masses)
                    or abs(fluid.total_mass - expected_mass) > max(1e-9, expected_mass * 1e-6)
                    or abs(current_time - index * dt) > 1e-8):
                raise RuntimeError("Particle count, mass, finite-state, or clock validation failed.")
            max_mass_error = max(max_mass_error, abs(float(masses.sum()) - model_mass))
            frames[index], times[index] = points, current_time
            metrics = geometry.classify_water(points, labels, current_time, spacing)
            max_below_table = max(max_below_table, int(metrics["below_table"]))
            if max_below_table:
                raise RuntimeError("A liquid particle passed below the tabletop.")
            hand = UsdGeom.Xformable(demo.rig.hand).ComputeLocalToWorldTransform(0)
            actual_tcp = np.asarray(hand.Transform((0, 0, geometry.PANDA_HAND_TO_TCP)))
            tube_origin, tube_quaternion = geometry.tube_pose(current_time)
            expected_tcp = tube_origin + geometry.tube_rotation(current_time) @ np.array(
                [0, 0, geometry.TUBE_LENGTH - geometry.TCP_BELOW_MOUTH])
            max_grasp_position = max(max_grasp_position, float(np.linalg.norm(actual_tcp - expected_tcp)))
            q = hand.ExtractRotationQuat()
            actual_hand_quaternion = (q.GetReal(), *q.GetImaginary())
            max_grasp_rotation = max(max_grasp_rotation, _quaternion_error(
                actual_hand_quaternion, geometry.hand_pose(current_time)[1]))
            collider = fluid.state_0.body_q.numpy()[fluid.tube_body]
            max_collider_position = max(max_collider_position, float(np.linalg.norm(collider[:3] - tube_origin)))
            max_collider_rotation = max(max_collider_rotation, _quaternion_error(
                collider[3:], tube_quaternion))
            demo.max_grasp_error = max(demo.max_grasp_error, max_grasp_position)
            if max_grasp_position >= 0.003 or max_grasp_rotation >= 0.03:
                raise RuntimeError("The robot hand lost its validated test-tube pose.")
            if max_collider_position >= 1e-6 or max_collider_rotation >= 1e-5:
                raise RuntimeError("The liquid collider no longer follows the test tube.")
            if 0.2 * base.MOTION_TIME_SCALE <= current_time <= 2.2 * base.MOTION_TIME_SCALE:
                local = (points[initial_tube] - tube_origin) @ geometry.tube_rotation(current_time)
                height = float(np.diff(np.quantile(local[:, 2], [0.05, 0.95]))[0])
                retention = metrics["tube_remaining"] / tube_count
                min_height = min(min_height, height / initial_height)
                min_retention = min(min_retention, retention)
                hold_samples += 1
                if min_height < 0.65 or min_retention < 0.95:
                    raise RuntimeError("Upright water failed its volume/column-holding check.")
            if index % 60 == 0 or index == steps:
                print(f"Baking pour: {current_time:.1f}/{duration:.1f} s; "
                      f"received {100 * metrics['tube_transfer_fraction']:.1f}%", flush=True)
            await app.next_update_async()
            _require_owned_scene(demo)
        _require_owned_scene(demo)
        report = demo.report()
        final = geometry.classify_water(frames[-1], labels, times[-1], spacing)
        checks = {
            "base_demo": bool(report["success"]),
            "fixed_particle_count_and_mass": True,
            "all_positions_finite": True,
            "upright_column": hold_samples > 0 and min_height >= 0.65 and min_retention >= 0.95,
            "tube_transfer": final["tube_transfer_fraction"] >= 0.75,
            "no_particles_below_table": max_below_table == 0,
            "robot_grasp": max_grasp_position < 0.003 and max_grasp_rotation < 0.03,
            "joint_velocity_limits": bool(report["joint_speeds_within_urdf_limits"]),
            "collider_pose": max_collider_position < 1e-6 and max_collider_rotation < 1e-5,
        }
        _require_owned_scene(demo)
        demo.reset()
        _require_owned_scene(demo)
        reset_error = float(np.max(np.abs(fluid.positions() - initial)))
        checks["reset"] = reset_error < 1e-7
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise RuntimeError("Cache validation failed: " + failed)
        validation = {
            "checks": checks,
            "particle_count": count,
            "total_mass_kg": model_mass,
            "max_mass_error_kg": max_mass_error,
            "min_upright_height_fraction": min_height,
            "min_upright_particle_retention": min_retention,
            "upright_samples": hold_samples,
            "tube_transfer_fraction": float(final["tube_transfer_fraction"]),
            "tube_spilled_particles": int(final["tube_spilled"]),
            "max_particles_below_table": max_below_table,
            "max_grasp_position_error_m": max_grasp_position,
            "max_grasp_orientation_error_rad": max_grasp_rotation,
            "max_collider_position_error_m": max_collider_position,
            "max_collider_orientation_error_rad": max_collider_rotation,
            "max_joint_velocity_limit_ratio": float(demo.rig.metrics["max_joint_velocity_limit_ratio"]),
            "reset_max_position_error_m": reset_error,
        }
        helper_path = root / "assets" / "robot_testtube_pour" / "scripts" / "liquid.py"
        metadata = {
            "schema_version": 1,
            "spacing": spacing,
            "dt": dt,
            "motion_time_scale": float(base.MOTION_TIME_SCALE),
            "mode": str(fluid.mode),
            "helper_revision": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
            "validation_success": True,
            "geometry": _geometry_metadata(geometry),
            "initialization": {
                "tube_fill_height": float(geometry.TUBE_FILL_HEIGHT),
                "beaker_water_depth": float(geometry.BEAKER_WATER_DEPTH),
                "density_kg_m3": 1000.0,
            },
            "motion_knots": [[float(t), [float(x) for x in p], float(a)]
                             for t, p, a in geometry.MOTION_KNOTS],
            "validation": validation,
        }
        _require_owned_scene(demo)
        _atomic_save(output, frames, times, labels, metadata,
                     require_current=lambda: _require_owned_scene(demo))
    finally:
        base.SURFACE_EVERY_N_STEPS = old_surface_interval
    try:
        cache_name = output.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        cache_name = output.name
    summary = {
        "success": True, "cache": cache_name,
        "frames": len(times), "fps": 60, "duration_seconds": duration,
        "particle_count": count, "mass_kg": model_mass,
        "received_fraction": validation["tube_transfer_fraction"],
        "min_upright_height_fraction": min_height,
        "min_upright_particle_retention": min_retention,
        "max_grasp_position_error_m": max_grasp_position,
        "reset_max_position_error_m": reset_error,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _start_bake_task():
    """Share the cancellation slot used by all of the project's demos."""
    import omni.kit.app

    try:
        current = asyncio.current_task()
    except RuntimeError:
        current = None
    previous_tasks = []
    for name in ("_robot_pour_build_task", "_fluid_lab_cache_bake_task"):
        previous = getattr(omni.kit.app, name, None)
        if (previous is not None and previous is not current
                and not any(previous is task for task in previous_tasks)):
            previous_tasks.append(previous)
            if not previous.done():
                previous.cancel()
    task = asyncio.ensure_future(bake())
    omni.kit.app._robot_pour_build_task = task
    omni.kit.app._fluid_lab_cache_bake_task = task
    return task


def _entrypoint():
    try:
        import omni.kit.app

        in_kit = omni.kit.app.get_app().is_running()
    except Exception:
        in_kit = False
    if in_kit:
        task = _start_bake_task()

        def completed(finished):
            try:
                finished.result()
            except asyncio.CancelledError:
                print("Liquid cache bake was cancelled; the previous cache was preserved.", flush=True)
            except Exception:
                import traceback

                traceback.print_exc()

        task.add_done_callback(completed)
        return task
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": "--headless" in sys.argv,
        "width": 1280, "height": 720, "renderer": "RaytracedLighting",
    })
    task = _start_bake_task()
    try:
        while simulation_app.is_running() and not task.done():
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise RuntimeError("Kit closed before the liquid cache was complete.")
        task.result()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _entrypoint()
