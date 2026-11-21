"""Visible 20/30/40 mL dosing with the frozen, live-physics-trained policy.

Run this file in Isaac Sim's Script Editor. The shared bounded robot path
remains kinematic; CEM learns the state-dependent stop/return decision.
No cached liquid, accelerated video time, or inferred camera volume is used.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT
import sys


INSTALL = PROJECT_ROOT
TARGET_VOLUME_ML = 30.0
WATER_STYLE = "training_blue"  # training_blue / clear, also switchable in the panel
USE_CUDA_GRAPH = False        # graph is batch-only: surface reconstruction interleaving is unvalidated
TABLE_STYLE = "white_matte"
POLICY_FILE = INSTALL / "assets" / "robot_testtube_pour" / "policy" / "target_volume_policy.json"
CANDIDATE_FILE = INSTALL / "assets" / "robot_testtube_pour" / "policy" / "target_volume_candidate.json"
OUTPUT = INSTALL / "droplet_out" / "robot_target_volume_ai"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


async def create_demo(attach=True):
    if not POLICY_FILE.is_file():
        raise FileNotFoundError("Train the live policy with train_target_volume_policy.py first: " + str(POLICY_FILE))
    learned = _load("visible_learned_pour_policy", INSTALL / "assets" / "robot_testtube_pour" /
                    "scripts" / "learned_policy.py")
    policy = learned.LearnedPourPolicy.load(POLICY_FILE)
    metadata = json.loads(POLICY_FILE.read_text(encoding="utf-8")).get("metadata", {})
    policy.selection_origin = metadata.get("selection_origin", metadata.get("selection", "unspecified"))
    policy.display_name = ("learned CEM feedback" if metadata.get("is_trained", True)
                           else "baseline selected after training")
    base = _load("visible_ai_target_pour", INSTALL / "robot_target_volume_pour.py")
    base.TARGET_VOLUME_ML = TARGET_VOLUME_ML
    base.WATER_STYLE, base.TABLE_STYLE = WATER_STYLE, TABLE_STYLE
    base.WINDOW_TITLE = "AI target volume pouring"
    base.VOLUME_TOLERANCE_ML = 0.25
    base.MAX_SIMULATION_SECONDS = 35.0
    base.OUTPUT = OUTPUT
    demo = await base.create_demo(attach=False, policy=policy)
    if USE_CUDA_GRAPH:
        graph = _load("visible_pbf_graph", INSTALL / "assets" / "robot_testtube_pour" /
                      "scripts" / "pbf_graph.py")
        demo.graph_dispatch = graph.install_graph_step(demo.fluid)
    demo.recommended_policy = policy
    selected_trained = bool(metadata.get("is_trained", True))
    demo.trained_policy = policy if selected_trained else None
    if CANDIDATE_FILE.is_file():
        candidate = learned.LearnedPourPolicy.load(CANDIDATE_FILE)
        candidate.selection_origin = "training_best_candidate"
        candidate.display_name = "learned CEM candidate"
        demo.trained_policy = candidate
    demo.learned_button_label = "Learned (validated)" if selected_trained else "Learned candidate"
    demo.baseline_button_label = "Baseline" if selected_trained else "Baseline (validated)"
    if attach:
        demo.attach()
        demo.show_controls()
    demo.policy_checkpoint = str(POLICY_FILE)
    print("Validated pouring policy loaded: " + str(policy.params) + "; " + policy.selection_origin, flush=True)
    return demo


def launch():
    import omni.kit.app
    request_epoch = getattr(omni.kit.app, "_water_demo_epoch", 0)
    async def build_once():
        if getattr(omni.kit.app, "_water_demo_epoch", 0) != request_epoch:
            raise asyncio.CancelledError("A newer water example was selected.")
        return await create_demo()
    previous = getattr(omni.kit.app, "_robot_pour_build_task", None)
    if previous is not None and not previous.done():
        previous.cancel()
    task = asyncio.ensure_future(build_once())
    omni.kit.app._robot_pour_build_task = task
    return task


def _entrypoint():
    try:
        import omni.kit.app
        if omni.kit.app.get_app().is_running():
            launch()
            return
    except Exception:
        pass
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
