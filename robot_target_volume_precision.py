"""Precision V2 dosing GUI, with the original baseline available for comparison.

This is manual closed-loop particle feedback with calibrated parameters, not
PPO or an end-to-end robot policy. Every action uses live PBF at physical time.
An accepted checkpoint must be supplied; missing files never select defaults.
"""

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT
import sys
from types import MethodType


# Script Editor runs a temporary buffer; use the explicit local project root.
INSTALL = PROJECT_ROOT
TARGET_VOLUME_ML = 30.0
WATER_STYLE = "training_blue"
TABLE_STYLE = "white_matte"
USE_CUDA_GRAPH = False
MAX_SIMULATION_SECONDS = 95.0
VOLUME_TOLERANCE_ML = 0.1
POLICY_FILE = INSTALL / "assets" / "robot_testtube_pour" / "policy" / "precision_v2.json"
OUTPUT = INSTALL / "droplet_out" / "robot_target_volume_precision"


def _load(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _load_policy(checkpoint_file=None):
    path = Path(POLICY_FILE if checkpoint_file is None else checkpoint_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Precision V2 checkpoint is unavailable. Finish calibration/validation or pass "
            "checkpoint_file explicitly; no default policy was substituted: " + str(path))
    directory = INSTALL / "assets" / "robot_testtube_pour" / "scripts"
    precision = _load("visible_precision_v2_control", directory / "precision_control.py")
    raw = path.read_bytes()
    metadata = json.loads(raw).get("metadata", {})
    policy = precision.PrecisionPolicy.load(path)
    policy.precision_calibrated = metadata.get("calibration") is True
    policy.display_name = ("Precision V2 calibrated closed-loop feedback" if policy.precision_calibrated
                           else "Precision V2 development parameters (uncalibrated)")
    policy.selection_origin = metadata.get("selection_origin", "explicit_precision_checkpoint")
    return path, policy, precision, hashlib.sha256(raw).hexdigest()


class _ControlsAdapter:
    """Dispatch None to the original controller and V2 policies to precision."""

    def __init__(self, baseline, precision):
        self.baseline = baseline
        self.precision = precision

    def TargetPourController(self, geometry, target_ml=30.0, spacing=0.0035, policy=None):
        controller = self.baseline.TargetPourController if policy is None else self.precision.TargetPourController
        return controller(geometry, target_ml=target_ml, spacing=spacing, policy=policy)


def _update_precision_label(demo):
    demo._precision_entry_state["original_update_label"]()
    mode = getattr(demo, "mode_label", None)
    if mode is not None:
        mode.text = (("Precision V2: calibrated feedback (not PPO)" if demo.policy.precision_calibrated
                      else "Precision V2: development parameters (not calibrated)")
                     if demo.policy is not None else "Original baseline: single-stop feedback")
    label = getattr(demo, "status_label", None)
    if label is not None:
        import omni.ui as ui
        label.height = ui.Pixel(92)
        pulse_count = int(demo.status.get("pulse_count", 0))
        failure = demo.status.get("precision_failure")
        label.text += f"\nFine pulses: {pulse_count}" + (f" | {failure}" if failure else "")
    import omni.kit.app
    window = getattr(omni.kit.app, "_robot_pour_window", None)
    if window is not None:
        window.height = max(window.height, 435)


def _precision_report(demo):
    result = demo._precision_entry_state["original_report"]()
    result.update({
        "precision_entrypoint": "robot_target_volume_precision.py",
        "control_mode": "precision_v2" if demo.policy is not None else "original_baseline",
        "control_description": "manual closed-loop particle feedback with parameter calibration; not PPO",
        "precision_checkpoint_file": demo.policy_checkpoint if demo.policy is not None else None,
        "precision_checkpoint_sha256": demo.policy_checkpoint_sha256 if demo.policy is not None else None,
        "cuda_graph": False,
        "pulse_count": int(demo.status.get("pulse_count", 0)),
        "precision_target_band_reached": demo.status.get("target_band_reached"),
        "precision_failure": demo.status.get("precision_failure"),
    })
    return result


def install_on_demo(demo, checkpoint_file=None, attach=True, target_ml=None):
    """Reset an existing owned TargetPourDemo without constructing another rig.

    This function is synchronous. Pass attach=False for an external validation
    loop; the existing timeline is paused before reset in either case.
    """
    if type(demo).__name__ != "TargetPourDemo":
        raise TypeError("Precision installation requires an existing TargetPourDemo.")
    demo._require_current()
    if demo.busy or getattr(demo, "_testing", False):
        raise RuntimeError("The owned pouring demo is busy; precision installation was not applied.")
    task = getattr(demo, "export_task", None)
    if task is not None and not task.done():
        raise RuntimeError("Finish or cancel the current demo task before replacing its controller.")
    if (getattr(demo.scene.rig, "original", None) is not demo.original_rig
            or type(demo.original_rig).__name__ == "_SourceRig"):
        raise RuntimeError("Expected one existing source-time rig wrapper; refusing nested wrapping.")
    if USE_CUDA_GRAPH:
        raise ValueError("Precision GUI uses original PBF dispatch; CUDA graph is not supported here.")
    path, policy, precision, checkpoint_sha = _load_policy(checkpoint_file)
    baseline = _load("visible_precision_original_control", INSTALL / "assets" /
                     "robot_testtube_pour" / "scripts" / "target_control.py")
    adapter = _ControlsAdapter(baseline, precision)

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    demo._require_current()
    if hasattr(demo.fluid, "_pbf_graph_step"):
        graph = _load("visible_precision_graph_helpers", INSTALL / "assets" /
                      "robot_testtube_pour" / "scripts" / "pbf_graph.py")
        graph.uninstall_graph_step(demo.fluid)
    demo.graph_dispatch = None

    state = getattr(demo, "_precision_entry_state", None)
    if state is None:
        state = {"original_update_label": demo._update_label, "original_report": demo.report}
        demo._precision_entry_state = state
    namespace = state["original_report"].__func__.__globals__
    namespace.update(MAX_SIMULATION_SECONDS=MAX_SIMULATION_SECONDS,
                     VOLUME_TOLERANCE_ML=VOLUME_TOLERANCE_ML,
                     OUTPUT=OUTPUT, WINDOW_TITLE="Precision V2 target volume pouring",
                     WATER_STYLE=WATER_STYLE)
    demo._update_label = MethodType(_update_precision_label, demo)
    demo.report = MethodType(_precision_report, demo)
    demo.controls_adapter = adapter
    demo.controls = adapter
    demo.policy = policy
    demo.precision_policy = policy
    demo.recommended_policy = policy
    demo.trained_policy = policy  # Existing panel's generic selected-policy slot.
    demo.learned_button_label = "Precision V2"
    demo.baseline_button_label = "Original baseline"
    demo.policy_checkpoint = str(path)
    demo.policy_checkpoint_sha256 = checkpoint_sha
    demo.reset(target_ml)
    demo.set_water_style(WATER_STYLE)
    timeline.set_current_time(0.0)
    omni.kit.app._robot_testtube_demo = demo
    if attach:
        demo.attach()
        demo.show_controls()
    print("Precision V2 checkpoint loaded: " + str(policy.params) + "; SHA256=" + checkpoint_sha,
          flush=True)
    return demo


async def create_demo(attach=True, checkpoint_file=None):
    """Construct a fresh live scene; use install_on_demo to preserve a scene."""
    _load_policy(checkpoint_file)  # Fail before any scene construction if absent/invalid.
    base = _load("visible_precision_target_pour", INSTALL / "robot_target_volume_pour.py")
    base.INSTALL = INSTALL
    base.TARGET_VOLUME_ML = TARGET_VOLUME_ML
    base.WATER_STYLE, base.TABLE_STYLE = WATER_STYLE, TABLE_STYLE
    base.WINDOW_TITLE = "Precision V2 target volume pouring"
    base.MAX_SIMULATION_SECONDS = MAX_SIMULATION_SECONDS
    base.VOLUME_TOLERANCE_ML = VOLUME_TOLERANCE_ML
    base.OUTPUT = OUTPUT
    demo = await base.create_demo(attach=False, policy=None)
    return install_on_demo(demo, checkpoint_file=checkpoint_file, attach=attach)


def launch(checkpoint_file=None):
    import omni.kit.app
    request_epoch = getattr(omni.kit.app, "_water_demo_epoch", 0)

    async def build_once():
        if getattr(omni.kit.app, "_water_demo_epoch", 0) != request_epoch:
            raise asyncio.CancelledError("A newer water example was selected.")
        existing = getattr(omni.kit.app, "_robot_testtube_demo", None)
        if (type(existing).__name__ == "TargetPourDemo" and existing._owns_scene()):
            return install_on_demo(existing, checkpoint_file=checkpoint_file)
        return await create_demo(checkpoint_file=checkpoint_file)

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
    app = SimulationApp({"headless": False, "width": 1280, "height": 720,
                         "renderer": "RaytracedLighting"})
    try:
        task = launch()
        while app.is_running():
            app.update()
            if task.done():
                task.result()
    finally:
        app.close()


if __name__ == "__main__":
    _entrypoint()
