"""Live feedback-controlled Franka pouring: select 20, 30 or 40 mL.

Open in Isaac Sim's Script Editor and Run. Select a target, then Play.
The fluid is solved live; the controller observes particle volumes and turns
the tube upright automatically. No recorded fluid playback is used here.
The pre-grasped tube and robot remain kinematic, as in the original example.
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
TARGET_VOLUME_ML = 30.0       # 20.0 / 30.0 / 40.0; also selectable in the panel
TARGET_OPTIONS_ML = (20.0, 30.0, 40.0)
VOLUME_TOLERANCE_ML = 2.0     # simulation acceptance tolerance, not hardware accuracy
TABLE_STYLE = "white_matte"
RENDER_MODE = "RealTimePathTracing"
SURFACE_EVERY_N_STEPS = 3
MAX_SIMULATION_SECONDS = 30.0
INSTALL = PROJECT_ROOT
OUTPUT = INSTALL / "droplet_out" / "robot_target_volume_pour"


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()),
                    encoding="utf-8")


class _SourceRig:
    """Use the same IK path parameter as the physically moving tube."""

    def __init__(self, rig, controller):
        self.original = rig
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.original, name)

    def update(self, physical_time):
        self.original.update(self.controller.source_time(physical_time))


class TargetPourDemo:
    def __init__(self, scene, base, controls, target_ml):
        self.scene, self.base, self.controls = scene, base, controls
        self._closed = False
        self.busy = False
        self._testing = False
        self._finished = False
        self.target_ml = float(target_ml)
        self.original_rig = scene.rig
        self.controller = controls.TargetPourController(
            scene.geometry, target_ml=self.target_ml, spacing=base.PARTICLE_SPACING)
        self.scene.rig = _SourceRig(self.original_rig, self.controller)
        self.fluid.pose_fn = self.controller.tube_pose
        self._initial_points = self.fluid.positions()
        self._initial_mass = self.fluid.total_mass
        self.reset()

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def _owns_scene(self):
        return not self._closed and self.scene._owns_scene()

    def _require_current(self):
        if not self._owns_scene():
            raise asyncio.CancelledError("The active water example changed.")

    def _observe(self):
        source = self.controller.source_time(self.fluid.time)
        positions = self.fluid.positions()
        metrics = self.geometry.classify_water(
            positions, self.fluid.labels, source, self.base.PARTICLE_SPACING)
        particle_ml = self.base.PARTICLE_SPACING ** 3 * 1e6
        metrics.update({"time": self.fluid.time, "source_coordinate": source,
                        "target_ml": self.target_ml,
                        "received_ml": metrics["tube_to_beaker"] * particle_ml,
                        "remaining_ml": metrics["tube_remaining"] * particle_ml,
                        "inflight_ml": metrics["tube_airborne"] * particle_ml,
                        "spilled_ml": metrics["tube_spilled"] * particle_ml})
        return metrics

    def _update_label(self):
        label = getattr(self, "status_label", None)
        if label is None:
            return
        m = self.current
        phase = self.status.get("phase", "ready")
        label.text = (f"Target: {self.target_ml:g} mL  |  Received: {m['received_ml']:.2f} mL\n"
                      f"{phase}  |  {self.fluid.time:.1f} s  |  "
                      f"In flight: {m['inflight_ml']:.2f} mL")

    def _grasp_error(self):
        from pxr import Gf, UsdGeom
        actual_hand = UsdGeom.Xformable(self.original_rig.hand).ComputeLocalToWorldTransform(0)
        tcp = actual_hand.Transform((0.0, 0.0, self.geometry.PANDA_HAND_TO_TCP))
        pose = self.fluid.state_0.body_q.numpy()[self.fluid.tube_body]
        actual_tube = self.base._pose_matrix(pose[:3], pose[3:])
        grasp = actual_tube.Transform((0.0, 0.0,
            self.geometry.TUBE_LENGTH - self.geometry.TCP_BELOW_MOUTH))
        q = actual_tube.ExtractRotationQuat() * Gf.Quatd(math.sqrt(.5), Gf.Vec3d(0, math.sqrt(.5), 0))
        h = actual_hand.ExtractRotationQuat()
        expected = np.array([q.GetReal(), *q.GetImaginary()])
        observed = np.array([h.GetReal(), *h.GetImaginary()])
        angle = float(2 * math.acos(np.clip(abs(np.dot(expected, observed)), 0.0, 1.0)))
        self.max_grasp_angle_error = max(self.max_grasp_angle_error, angle)
        return float(np.linalg.norm(np.asarray(tcp) - np.asarray(grasp)))

    def reset(self, target_ml=None):
        self._require_current()
        if target_ml is not None:
            target_ml = float(target_ml)
            if target_ml not in TARGET_OPTIONS_ML:
                raise ValueError(f"Choose one of {TARGET_OPTIONS_ML} mL.")
            self.target_ml = target_ml
        self.controller = self.controls.TargetPourController(
            self.geometry, target_ml=self.target_ml, spacing=self.base.PARTICLE_SPACING)
        self.scene.rig.controller = self.controller
        self.fluid.pose_fn = self.controller.tube_pose
        self.fluid.reset()
        self.scene.step_count = 0
        self.status = self.controller.update(0.0, self.fluid.positions(), self.fluid.labels,
                                             self.fluid.state_0.particle_qd.numpy())
        self.current = self._observe()
        self.history = [dict(self.current)]
        self.max_grasp_error = 0.0
        self.max_grasp_angle_error = 0.0
        self.max_source_rate = 0.0
        self._last_source = 0.0
        self.max_below_table = 0
        self._finished = False
        self._last_phase = self.status.get("phase")
        self.scene.sync_visuals(force_surface=True)
        self._update_label()

    def advance(self, render=True):
        self._require_current()
        if self._finished:
            return False
        self.fluid.step_frame()
        self.scene.step_count += 1
        self.status = self.controller.update(
            self.fluid.time, self.fluid.positions(), self.fluid.labels,
            self.fluid.state_0.particle_qd.numpy())
        self.current = self._observe()
        self.max_source_rate = max(self.max_source_rate,
                                  abs(self.current["source_coordinate"] - self._last_source) / self.fluid.dt)
        self._last_source = self.current["source_coordinate"]
        if self.current["total"] != len(self._initial_points):
            raise RuntimeError("The fixed particle count changed.")
        self.max_below_table = max(self.max_below_table, self.current["below_table"])
        if render:
            self.scene.sync_visuals()
        else:
            self.scene.rig.update(self.fluid.time)
        self.max_grasp_error = max(self.max_grasp_error, self._grasp_error())
        phase = self.status.get("phase", "unknown")
        if phase != self._last_phase or self.scene.step_count % 15 == 0:
            self.history.append({**self.current, "controller_phase": phase})
            self._last_phase = phase
            self._update_label()
        if self.status.get("done", False) or self.fluid.time >= MAX_SIMULATION_SECONDS:
            self._finished = True
            self.scene.sync_visuals(force_surface=True)
            self.max_grasp_error = max(self.max_grasp_error, self._grasp_error())
            self._update_label()
            result = self.save_report()
            print(f"Target pour: {self.target_ml:g} mL -> {result['received_ml']:.2f} mL; "
                  f"error {result['error_ml']:+.2f} mL; success={result['success']}", flush=True)
        return True

    def report(self):
        m = self._observe()
        source = self.controller.source_time(self.fluid.time)
        tilt = abs(math.degrees(self.geometry.mouth_pose(source)[1]))
        error = m["received_ml"] - self.target_ml
        reset_error = getattr(self, "last_reset_error_m", 0.0)
        passed = (bool(self.status.get("done", False))
                  and abs(error) <= VOLUME_TOLERANCE_ML and tilt < .1
                  and self.max_below_table == 0 and m["spilled_ml"] <= 1.0
                  and self.max_grasp_error < .003 and self.max_grasp_angle_error < .02
                  and self.max_source_rate <= 1.00001
                  and self.fluid.total_mass == self._initial_mass
                  and reset_error < 1e-7)
        return {"success": passed, "complete": self._finished,
                "target_ml": self.target_ml, "received_ml": m["received_ml"],
                "error_ml": error, "tolerance_ml": VOLUME_TOLERANCE_ML,
                "remaining_ml": m["remaining_ml"], "spilled_ml": m["spilled_ml"],
                "inflight_ml": m["inflight_ml"], "final_tilt_degrees": tilt,
                "simulation_seconds": self.fluid.time,
                "wall_seconds": time.perf_counter() - getattr(self, "_run_started", time.perf_counter()),
                "particle_count": self.fluid.particle_count,
                "particle_volume_ml": self.base.PARTICLE_SPACING ** 3 * 1e6,
                "mass_conserved": self.fluid.total_mass == self._initial_mass,
                "max_below_table_particles": self.max_below_table,
                "max_grasp_position_error_m": self.max_grasp_error,
                "max_grasp_orientation_error_rad": self.max_grasp_angle_error,
                "max_source_rate": self.max_source_rate,
                "source_bank_joint_velocity_limit_ratio": self.original_rig.metrics["max_joint_velocity_limit_ratio"],
                "reset_max_position_error_m": reset_error,
                "fluid_mode": "live_PBF_feedback_control",
                "volume_measurement": "initial_tube_particle_ids_inside_beaker",
                "contact_force_grasp_simulated": False,
                "controller": self.status, "history": self.history}

    def save_report(self):
        result = self.report()
        _write_json(OUTPUT / f"target_{self.target_ml:g}ml.json", result)
        return result

    def attach(self):
        import omni.kit.app
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0.0)
        timeline.set_end_time(MAX_SIMULATION_SECONDS + 1.0)
        timeline.set_time_codes_per_second(self.base.SIMULATION_FPS)
        timeline.set_looping(False)
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(MAX_SIMULATION_SECONDS * self.base.SIMULATION_FPS)

        def on_update(_event):
            if not self._owns_scene() or self.busy or not timeline.is_playing():
                return
            self.busy = True
            try:
                self.advance()
                timeline.set_current_time(self.fluid.time)
                if self._finished:
                    timeline.pause()
            except Exception:
                timeline.pause()
                raise
            finally:
                self.busy = False

        def on_timeline(event):
            if not self._owns_scene():
                return
            kind = int(event.type)
            if kind == int(omni.timeline.TimelineEventType.STOP):
                if self._testing:
                    self.cancel_test()
                else:
                    self.reset()
            elif self._testing:
                return
            elif kind == int(omni.timeline.TimelineEventType.PLAY):
                if self._finished:
                    timeline.set_current_time(0.0)
                    self.reset()
                self._run_started = time.perf_counter()

        omni.kit.app._robot_pour_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            on_update, name="target volume live pouring")
        omni.kit.app._robot_pour_timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            on_timeline, name="target volume reset")

    def select_target(self, target):
        if not self._owns_scene() or self.busy:
            return
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(0.0)
        self.reset(target)

    def stop(self):
        if not self._owns_scene():
            return
        if self._testing:
            self.cancel_test()
            return
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(0.0)
        self.reset()

    def show_controls(self):
        import omni.kit.app
        import omni.timeline
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        timeline = omni.timeline.get_timeline_interface()

        def view(name):
            if self._owns_scene():
                viewport = get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = self.cameras[name]

        window = ui.Window("Target volume pouring", width=365, height=275,
                           position_x=70, position_y=610)
        with window.frame:
            with ui.VStack(spacing=7):
                ui.Label("Franka - live volume feedback", height=24)
                self.status_label = ui.Label("", height=42)
                with ui.HStack(height=28, spacing=5):
                    for target in TARGET_OPTIONS_ML:
                        ui.Button(f"{target:g} mL", clicked_fn=lambda value=target: self.select_target(value))
                with ui.HStack(height=28, spacing=5):
                    ui.Button("Play", clicked_fn=timeline.play)
                    ui.Button("Pause", clicked_fn=timeline.pause)
                    ui.Button("Reset", clicked_fn=self.stop)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("Overview", clicked_fn=lambda: view("overview"))
                    ui.Button("Pour close-up", clicked_fn=lambda: view("close"))
                with ui.HStack(height=28, spacing=5):
                    ui.Button("Test 20 / 30 / 40 mL", clicked_fn=self.test_all)
                    ui.Button("Cancel test", clicked_fn=self.cancel_test)
                ui.Label("Live fluid; pre-grasped kinematic tube", height=20)
        omni.kit.app._robot_pour_window = window
        self._update_label()

    def cancel_test(self):
        task = getattr(self, "export_task", None)
        if task is not None and not task.done():
            task.cancel()

    def test_all(self):
        previous = getattr(self, "export_task", None)
        if previous is not None and not previous.done():
            return previous
        if not self._owns_scene() or self.busy:
            return None
        self.export_task = asyncio.ensure_future(self.validate_all())
        return self.export_task

    async def validate_all(self, capture=True):
        import omni.kit.app
        import omni.timeline
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        self._require_current()
        if self._testing:
            raise RuntimeError("A target-volume test is already running.")
        timeline = omni.timeline.get_timeline_interface()
        self.busy, self._testing = True, True
        # Establish an active paused timeline, so native Stop emits STOP even
        # when the sweep was launched directly from an initially stopped stage.
        timeline.play()
        timeline.pause()
        selected = self.target_ml
        results = []
        started = time.perf_counter()
        viewport = get_active_viewport()
        try:
            for target in TARGET_OPTIONS_ML:
                self._require_current()
                self.reset(target)
                self.last_reset_error_m = float(np.max(np.abs(self.fluid.positions() - self._initial_points)))
                self._run_started = time.perf_counter()
                while not self._finished:
                    self._require_current()
                    # Solve every physical frame, but render fewer during the sweep.
                    for _ in range(3):
                        self.advance(render=False)
                        if self._finished:
                            break
                    self.scene.sync_visuals(force_surface=True)
                    self.max_grasp_error = max(self.max_grasp_error, self._grasp_error())
                    self._update_label()
                    await omni.kit.app.get_app().next_update_async()
                self._require_current()
                if capture and viewport is not None:
                    viewport.camera_path = self.cameras["overview"]
                    for _ in range(5):
                        await omni.kit.app.get_app().next_update_async()
                        self._require_current()
                    await capture_viewport_to_file(viewport, file_path=str(
                        OUTPUT / f"target_{target:g}ml.png")).wait_for_result()
                    self._require_current()
                results.append(self.save_report())
                _write_json(OUTPUT / "validation.json", {"complete": False, "results": results})
            report = {"complete": True, "success": all(r["success"] for r in results),
                      "elapsed_seconds": time.perf_counter() - started, "results": results}
            _write_json(OUTPUT / "validation.json", report)
            print("Target-volume sweep: " + json.dumps([
                {k: r[k] for k in ("target_ml", "received_ml", "error_ml", "success")}
                for r in results]), flush=True)
            return report
        finally:
            self.busy, self._testing = False, False
            if self._owns_scene():
                timeline.pause()
                timeline.set_current_time(0.0)
                self.reset(selected)
                if getattr(self, "status_label", None) is not None and len(results) == len(TARGET_OPTIONS_ML):
                    self.status_label.text = "Tests finished - select target, then Play\n" + " / ".join(
                        f"{r['target_ml']:g}: {r['received_ml']:.2f} mL" for r in results)


async def create_demo(attach=True):
    import omni.kit.app
    if float(TARGET_VOLUME_ML) not in TARGET_OPTIONS_ML:
        raise ValueError(f"TARGET_VOLUME_ML must be one of {TARGET_OPTIONS_ML}.")
    base = _load_source("target_pour_base", INSTALL / "robot_testtube_pour.py")
    base.LIQUID_MODE = "simulate"
    base.SHOW_CONTROL_PANEL = False
    base.SURFACE_EVERY_N_STEPS = SURFACE_EVERY_N_STEPS
    base.TABLE_STYLE, base.RENDER_MODE = TABLE_STYLE, RENDER_MODE
    scene = await base.create_demo(attach=False)
    controls = _load_source("target_pour_control", INSTALL / "assets" / "robot_testtube_pour" /
                            "scripts" / "target_control.py")
    demo = TargetPourDemo(scene, base, controls, TARGET_VOLUME_ML)
    demo._require_current()
    omni.kit.app._robot_testtube_demo = demo
    if attach:
        demo.attach()
        demo.show_controls()
    print(f"Target-volume pour ready: {TARGET_VOLUME_ML:g} mL. Select 20 / 30 / 40 mL, then Play.", flush=True)
    return demo


def launch():
    import omni.kit.app
    request_epoch = getattr(omni.kit.app, "_water_demo_epoch", 0)

    async def build_once():
        if getattr(omni.kit.app, "_water_demo_epoch", 0) != request_epoch:
            raise asyncio.CancelledError("A newer water example was run.")
        try:
            return await create_demo()
        except Exception as error:
            if getattr(omni.kit.app, "_robot_pour_build_task", None) is asyncio.current_task():
                import traceback
                _write_json(OUTPUT / "build_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            raise

    previous = getattr(omni.kit.app, "_robot_pour_build_task", None)
    if previous is not None and not previous.done():
        previous.cancel()
    task = asyncio.ensure_future(build_once())
    omni.kit.app._robot_pour_build_task = task
    return task


def _entrypoint():
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
    app = SimulationApp({"headless": test, "width": 1280, "height": 720,
                         "renderer": "RaytracedLighting"})
    try:
        task = asyncio.ensure_future(create_demo(attach=not test))
        while app.is_running() and not task.done():
            app.update()
        demo = task.result()
        if test:
            task = asyncio.ensure_future(demo.validate_all())
            while app.is_running() and not task.done():
                app.update()
            if not task.result()["success"]:
                raise RuntimeError("Target-volume validation failed; inspect validation.json.")
        else:
            while app.is_running():
                app.update()
    finally:
        app.close()


if __name__ == "__main__":
    _entrypoint()
