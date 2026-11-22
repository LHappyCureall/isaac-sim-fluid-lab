"""Publish an explicit source allowlist without changing the local experiments.

The Windows workspace default is a convenience for the author's daily sync.
The published examples use lab_paths instead. Every transformation has a fixed
anchor count; any mismatch aborts before destination files are changed.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path(r"C:\omninverse")
FULL_MODEL_SOURCES = {
    "droplet_out/pour_policy_training/run_20261122_full/candidate_frozen.json":
        ("assets/robot_testtube_pour/policy/full_20261122_candidate.json", "candidate"),
    "droplet_out/pour_policy_training/run_20261122_full/recommended_frozen.json":
        ("assets/robot_testtube_pour/policy/full_20261122_recommended.json", "recommended"),
}
PRECISION_RUN = "droplet_out/pour_policy_precision_v2/run_20261122_calibrated"
PRECISION_MODEL_SOURCES = {
    PRECISION_RUN + "/candidate_frozen.json":
        ("assets/robot_testtube_pour/policy/precision_v2_20261122_candidate.json", "candidate"),
    PRECISION_RUN + "/recommended_frozen.json":
        ("assets/robot_testtube_pour/policy/precision_v2_20261122_recommended.json", "recommended"),
    "assets/robot_testtube_pour/policy/precision_v2.json":
        ("assets/robot_testtube_pour/policy/precision_v2.json", "recommended"),
}
PRECISION_CONFIG_SOURCE = "projects/pour_policy_training_v2/calibration_configs.json"
FAST_RUN = "droplet_out/pour_policy_fast_v3/run_20261122_cem5s"
FAST_PROJECT = "projects/pour_policy_training_v3/"
FAST_TOOL_SOURCES = {FAST_PROJECT + name: "tools/training/" + name for name in (
    "prepare_fast_path.py", "prepare_smooth_fast_path.py", "verify_fast_path.py",
    "export_owned_kinematic_bank.py", "pilot_fast.py", "audit_fast_run.py", "plot_fast_results.py")}
FAST_RUNTIME_SOURCES = ("train_target_volume_fast.py", *["assets/robot_testtube_pour/scripts/" + name
    for name in ("fast_geometry.py", "fast_control.py", "fast_training_env.py")])
FAST_MODEL_SOURCES = {
    FAST_RUN + "/candidate_frozen.json":
        ("assets/robot_testtube_pour/policy/fast_v3_20261122_candidate.json", "candidate"),
    FAST_RUN + "/recommended_frozen.json":
        ("assets/robot_testtube_pour/policy/fast_v3_20261122_recommended.json", "recommended"),
}
FAST_PARAMETER_BOUNDS = {
    "transport_rate_fraction": (0.6, 1.0), "fast_tilt_deg_s": (45.0, 140.0),
    "meter_tilt_deg_s": (10.0, 80.0), "return_tilt_deg_s": (70.0, 180.0),
    "tail_lead_s": (0.02, 0.8), "stop_bias_ml": (-5.0, 10.0),
}
SOURCE_TO_DEST = {
    "projects/pour_policy_training/audit_candidate.py": "tools/training/audit_candidate.py",
    "projects/pour_policy_training/plot_results.py": "tools/training/plot_results.py",
    "projects/pour_policy_training/plot_full_results.py": "tools/training/plot_full_results.py",
    "projects/pour_policy_training_v2/audit_precision_run.py": "tools/training/audit_precision_run.py",
    "projects/pour_policy_training_v2/plot_precision_results.py": "tools/training/plot_precision_results.py",
    PRECISION_CONFIG_SOURCE: "configs/precision_v2_calibration.json",
    **{source: item[0] for source, item in FULL_MODEL_SOURCES.items()},
    **{source: item[0] for source, item in PRECISION_MODEL_SOURCES.items()},
    **FAST_TOOL_SOURCES,
    **{source: item[0] for source, item in FAST_MODEL_SOURCES.items()},
}
MODEL_FILES = (
    "assets/robot_testtube_pour/policy/target_volume_policy.json",
    "assets/robot_testtube_pour/policy/target_volume_candidate.json",
)
CHECKPOINT_SOURCES = (*MODEL_FILES, *FULL_MODEL_SOURCES, *PRECISION_MODEL_SOURCES, *FAST_MODEL_SOURCES)
ALLOWLIST = (
    "droplet_milkcrown_newton.py",
    "droplet_milkcrown_newton_surface.py",
    "droplet_milkcrown_newton_view.py",
    "robot_testtube_pour.py",
    "robot_target_volume_pour.py",
    "robot_target_volume_ai.py",
    "robot_target_volume_precision.py",
    "train_target_volume_policy.py",
    "train_target_volume_policy_full.py",
    "train_target_volume_precision.py",
    *FAST_RUNTIME_SOURCES,
    "assets/robot_testtube_pour/scripts/geometry.py",
    "assets/robot_testtube_pour/scripts/robot.py",
    "assets/robot_testtube_pour/scripts/liquid.py",
    "assets/robot_testtube_pour/scripts/target_control.py",
    "assets/robot_testtube_pour/scripts/volume_visuals.py",
    "assets/robot_testtube_pour/scripts/pour_training_env.py",
    "assets/robot_testtube_pour/scripts/learned_policy.py",
    "assets/robot_testtube_pour/scripts/pbf_graph.py",
    "assets/robot_testtube_pour/scripts/precision_control.py",
    "assets/robot_testtube_pour/scripts/precision_training_env.py",
    "assets/robot_testtube_pour/scripts/video.py",
    "assets/robot_testtube_pour/README.md",
    "assets/robot_testtube_pour/TARGET_VOLUME.md",
    "assets/robot_testtube_pour/AI_TRAINING.md",
    "textures/table_options/sources.json",
    *SOURCE_TO_DEST,
    *MODEL_FILES,
)


class SyncError(RuntimeError):
    pass


def _replace(text, old, new, *, count=1):
    found = text.count(old)
    if found != count:
        raise SyncError(f"Anchor {old!r}: expected {count} occurrence(s), found {found}.")
    return text.replace(old, new)


def _validate_checkpoint(relative, text):
    """Validate the reviewed CEM or distinct precision checkpoint schema."""
    fast_role = next((role for source, (destination, role) in FAST_MODEL_SOURCES.items()
                      if relative in (source, destination)), None)
    if fast_role is not None:
        return _validate_fast_checkpoint(text, fast_role)
    precision_role = next((role for source, (destination, role) in PRECISION_MODEL_SOURCES.items()
                           if relative in (source, destination)), None)
    if precision_role is not None:
        return _validate_precision_checkpoint(text, precision_role)
    if len(text.encode("utf-8")) > 64 * 1024:
        raise SyncError(f"Unexpectedly large policy checkpoint: {relative}.")
    value = json.loads(text)
    expected = {"schema_version", "policy_kind", "parameters", "observation_units", "target_feature", "metadata"}
    if not isinstance(value, dict) or set(value) != expected:
        raise SyncError("Policy checkpoint fields differ from the reviewed schema.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SyncError("Unsupported policy checkpoint schema version.")
    if value["policy_kind"] != "cem_linear_tail_compensation_v1":
        raise SyncError("Only the reviewed three-parameter CEM feedback policy can publish.")
    bounds = {"bias_ml": (-0.8, 0.8), "lead_seconds": (0.0, 0.5), "target_slope_ml": (-0.5, 0.5)}
    parameters = value["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(bounds):
        raise SyncError("Policy must contain exactly the three reviewed parameters.")
    for name, (low, high) in bounds.items():
        parameter = parameters[name]
        if (type(parameter) not in (int, float) or not low <= parameter <= high
                or not math.isfinite(parameter)):
            raise SyncError(f"Nonfinite, nonnumeric or out-of-bounds policy parameter: {name}.")
    if value["observation_units"] != {"volume": "mL", "flow": "mL/s", "tilt": "degrees"}:
        raise SyncError("Policy observation units changed.")
    if value["target_feature"] != "(target_ml - 30.0) / 10.0":
        raise SyncError("Policy target-feature normalization changed.")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or type(metadata.get("is_trained")) is not bool:
        raise SyncError("Policy metadata must explicitly identify whether parameters were trained.")
    full_role = next((role for source, (destination, role) in FULL_MODEL_SOURCES.items()
                      if relative in (source, destination)), None)
    if full_role is not None:
        if (metadata.get("policy_role") != full_role or metadata.get("test_used_for_selection") is not False
                or metadata.get("gui_deployment_written") is not False):
            raise SyncError("Full-run model role/isolation metadata changed.")
        baseline = {"bias_ml": 0.2, "lead_seconds": 0.2, "target_slope_ml": 0.0}
        if full_role == "candidate":
            changed = parameters != baseline
            if (metadata.get("selection_origin") != "full_CEM_training_best"
                    or metadata.get("parameters_changed_from_baseline") is not changed
                    or metadata["is_trained"] is not changed):
                raise SyncError("Full candidate must identify whether search changed baseline parameters.")
        else:
            origin = metadata.get("selection_origin")
            if origin not in ("baseline_fallback", "cem_training_best") or metadata["is_trained"] != (origin == "cem_training_best"):
                raise SyncError("Full recommendation training label contradicts validation selection.")
            if origin == "baseline_fallback" and parameters != baseline:
                raise SyncError("Full baseline recommendation must retain baseline parameters.")
    elif relative == MODEL_FILES[0]:
        origin = metadata.get("selection_origin", metadata.get("selection"))
        if origin not in ("baseline_fallback", "cem_training_best"):
            raise SyncError("Default policy must identify its validation-based selection origin.")
        if metadata["is_trained"] != (origin == "cem_training_best"):
            raise SyncError("Default policy training label contradicts its selection origin.")
        if origin == "baseline_fallback" and parameters != {"bias_ml": 0.2, "lead_seconds": 0.2, "target_slope_ml": 0.0}:
            raise SyncError("A baseline fallback must preserve the original baseline parameters.")
    elif (metadata["is_trained"] is not True or metadata.get("default_deployment") is not False
          or metadata.get("selection_origin") != "research_candidate_not_default_deployment"):
        raise SyncError("Candidate checkpoint must be explicitly marked trained and not the default deployment.")
    # Also reject NaN/Infinity hidden in metadata; json.loads accepts these by
    # default even though they are not valid interoperable JSON numbers.
    json.dumps(value, allow_nan=False)


def _validate_precision_parameters(parameters):
    bounds = {"reserve_ml": (1.5, 4.0), "probe_rate": (0.001, 0.010),
              "pulse_particles": (1, 3), "retract_source": (0.02, 0.08),
              "fine_retract_source": (0.004, 0.02)}
    if not isinstance(parameters, dict) or set(parameters) != set(bounds):
        raise SyncError("Precision policy requires exactly its five reviewed parameters.")
    for name, (low, high) in bounds.items():
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise SyncError("Invalid precision-policy parameter: " + name)
        if name == "pulse_particles" and type(value) is not int:
            raise SyncError("Precision pulse_particles must be an integer.")


def _validate_precision_checkpoint(text, role):
    if len(text.encode("utf-8")) > 64 * 1024:
        raise SyncError("Unexpectedly large precision-policy checkpoint.")
    value = json.loads(text)
    fields = {"schema_version", "policy_kind", "parameters", "observation_units", "termination", "metadata"}
    if (not isinstance(value, dict) or set(value) != fields
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["policy_kind"] != "closed_loop_particle_pulse_precision_v2"):
        raise SyncError("Unsupported precision-policy schema or kind; old CEM weights are not interchangeable.")
    _validate_precision_parameters(value["parameters"])
    if (value["observation_units"] != {"volume": "mL", "flow": "mL/s", "tilt": "degrees"}
            or value["termination"] != "nearest_requested_volume_particle_count_plus_or_minus_one"):
        raise SyncError("Precision-policy units or particle-band semantics changed.")
    metadata = value["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("role") != role
            or metadata.get("algorithm") != "hand_designed_coarse_to_fine_feedback_plus_predeclared_finite_local_calibration"
            or metadata.get("neural_policy") is not False or metadata.get("ppo") is not False
            or metadata.get("calibration") is not True or metadata.get("test_used_for_selection") is not False
            or metadata.get("automatic_gui_deployment") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("protocol_id", "")))):
        raise SyncError("Precision calibration/role/isolation metadata differs from the reviewed protocol.")
    json.dumps(value, allow_nan=False)


def _validate_fast_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(FAST_PARAMETER_BOUNDS):
        raise SyncError("Fast V3 requires exactly six distinct speed/stop parameters.")
    for name, (low, high) in FAST_PARAMETER_BOUNDS.items():
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise SyncError("Invalid fast V3 parameter: " + name)


def _validate_fast_checkpoint(text, role):
    if len(text.encode("utf-8")) > 64 * 1024:
        raise SyncError("Unexpectedly large fast V3 research checkpoint.")
    value = json.loads(text)
    fields = {"schema_version", "policy_kind", "parameters", "observation_units", "action_units",
              "deadline_physical_seconds", "metadata"}
    if (not isinstance(value, dict) or set(value) != fields or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["policy_kind"] != "joint_time_fast_pouring_v3"):
        raise SyncError("Fast V3 has its own six-parameter schema; V1/V2 weights are not interchangeable.")
    _validate_fast_parameters(value["parameters"])
    if (value["observation_units"] != {"volume": "mL", "flow": "mL/s", "tilt": "degrees", "physical_time": "seconds"}
            or value["action_units"] != {"tilt_rates": "degrees/physical-second", "u_rate": "joint-time-u/physical-second"}
            or value["deadline_physical_seconds"] != 5.0):
        raise SyncError("Fast V3 physical units/deadline changed.")
    metadata = value["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("role") != role
            or metadata.get("algorithm") != "feasibility_first_CEM_speed_and_stop_feedback_v3"
            or metadata.get("neural_policy") is not False or metadata.get("ppo") is not False
            or metadata.get("training_episodes") != 108 or metadata.get("test_used_for_selection") is not False
            or metadata.get("automatic_gui_deployment") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("protocol_id", "")))
            or not isinstance(metadata.get("source_sha256"), dict) or len(metadata["source_sha256"]) != 20):
        raise SyncError("Fast V3 role, completed training, source or isolation metadata differs.")
    json.dumps(value, allow_nan=False)


def _transform_fast_source(relative, text):
    """Portable research tools; retain all solver/controller calculations."""
    cache_dir = 'ROOT / "_cache/pour_training/fast_v3"'
    if relative == "train_target_volume_fast.py":
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT, require_isaac_sim_root\n")
        text = _replace(text, "ROOT = Path(__file__).resolve().parent", "ROOT = PROJECT_ROOT")
        text = _replace(text, 'sdk = Path(os.environ.get("ISAAC_SIM_ROOT", str(ROOT)))',
                        "sdk = require_isaac_sim_root()", count=2)
        for name in ("prepare_fast_path.py", "prepare_smooth_fast_path.py", "verify_fast_path.py", "export_owned_kinematic_bank.py"):
            text = _replace(text, FAST_PROJECT + name, FAST_TOOL_SOURCES[FAST_PROJECT + name])
        text = _replace(text, 'home = ROOT / "projects/pour_policy_training_v3"', "home = " + cache_dir)
        text = _replace(text, 'home / "prototype_v001/fast_path_cache.npz"', 'home / "fast_path_cache.npz"')
        text = _replace(text, 'home / "prototype_v001/fast_path_cache.json"', 'home / "fast_path_cache.json"')
    elif relative.endswith("/fast_geometry.py"):
        text = _replace(text, "import math\n", "import math\nimport os\n")
        text = _replace(text, 'PROJECT_ROOT / "projects/pour_policy_training_v3/fast_path_smooth_cache.npz"',
                        'PROJECT_ROOT / "_cache/pour_training/fast_v3/fast_path_smooth_cache.npz"')
        text = _replace(text, 'CACHE_SHA256 = "9d4e24e28e93d21320856ed7ae67282b32d8ce62587d3e46131b2d9a8bcbe7ae"',
                        'CACHE_SHA256 = os.environ.get("ISAAC_FAST_PATH_CACHE_SHA256", "")\n'
                        'if len(CACHE_SHA256) != 64 or any(c not in "0123456789abcdef" for c in CACHE_SHA256):\n'
                        '    raise RuntimeError("Rebuild and CPU-verify the local robot path cache, then explicitly set ISAAC_FAST_PATH_CACHE_SHA256 to its new SHA256. The archived research cache is not distributed.")')
    elif relative.endswith("/fast_control.py") or relative.endswith("/fast_training_env.py"):
        anchor = ('"""Five-physical-second pouring on a joint-time-parameterized existing path.'
                  if relative.endswith("/fast_control.py") else
                  '"""Five-second full-cycle collector using unchanged live PBF and fixed mass.')
        _replace(text, anchor, anchor)
    elif relative in FAST_TOOL_SOURCES:
        name = Path(relative).name
        if name == "prepare_fast_path.py":
            text = _replace(text, "import math\n", "import math\nimport sys\n")
            text = _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                            'ROOT = Path(__file__).resolve().parents[2]\n'
                            'sys.path.insert(0, str(ROOT))\nfrom lab_paths import require_isaac_sim_root')
            text = _replace(text, "HERE = Path(__file__).resolve().parent", "HERE = " + cache_dir)
            text = _replace(text, 'URDF = ROOT / "exts/', 'URDF = require_isaac_sim_root() / "exts/')
            text = _replace(text, '    if output.exists() or output.with_suffix(".json").exists():',
                            '    output.parent.mkdir(parents=True, exist_ok=True)\n'
                            '    if output.exists() or output.with_suffix(".json").exists():')
        elif name == "prepare_smooth_fast_path.py":
            text = _replace(text, "DIRECTORY = Path(__file__).resolve().parent",
                            'ROOT = Path(__file__).resolve().parents[2]\nDIRECTORY = ' + cache_dir)
            text = _replace(text, 'DIRECTORY / "prototype_v001/fast_path_cache.npz"', 'DIRECTORY / "fast_path_cache.npz"')
            text = _replace(text, "def prepare():\n", 'def prepare():\n'
                            '    DIRECTORY.mkdir(parents=True, exist_ok=True)\n'
                            '    if OUTPUT.exists() or OUTPUT.with_suffix(".json").exists():\n'
                            '        raise FileExistsError("Preserve existing robot path evidence; use a fresh local cache directory.")\n')
        elif name == "verify_fast_path.py":
            text = _replace(text, "DIRECTORY = Path(__file__).resolve().parent", "DIRECTORY = " + cache_dir)
        elif name == "pilot_fast.py":
            text = _replace(text, "ROOT = Path(r'C:\\omninverse')",
                            'ROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0, str(ROOT))')
        elif name == "audit_fast_run.py":
            text = _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                            'ROOT = Path(__file__).resolve().parents[2]\n'
                            'sys.path.insert(0, str(ROOT))\nfrom lab_paths import require_isaac_sim_root')
            text = _replace(text, 'sdk = Path(os.environ.get("ISAAC_SIM_ROOT", str(root)))', 'sdk = require_isaac_sim_root()')
        elif name == "plot_fast_results.py":
            _replace(text, "ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path(__file__).resolve().parents[2]")
        elif name == "export_owned_kinematic_bank.py":
            text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\nimport sys\n")
            text = _replace(text, "demo = omni.kit.app._robot_testtube_demo\n",
                            'ROOT = Path(__file__).resolve().parents[2]\n'
                            'sys.path.insert(0, str(ROOT))\n'
                            'demo = getattr(omni.kit.app, "_robot_testtube_demo", None)\n'
                            'if demo is None or type(demo).__name__ != "TargetPourDemo" or getattr(demo, "policy", None) is not None:\n'
                            '    raise RuntimeError("Open this repository baseline target-volume demo and pause it before this read-only export.")\n'
                            'if (Path(demo.base.__file__).resolve() != ROOT / "robot_testtube_pour.py"\n'
                            '        or Path(demo.geometry.__file__).resolve() != ROOT / "assets/robot_testtube_pour/scripts/geometry.py"\n'
                            '        or not demo.scene._owns_scene()):\n'
                            '    raise RuntimeError("Refusing to export an unknown or stale scene.")\n'
                            'if not np.allclose([k[0] for k in demo.geometry.MOTION_KNOTS[:5]], [0., 1.4, 4.4, 6.4, 8.4], atol=1e-10):\n'
                            '    raise RuntimeError("Expected exactly one legacy source-time scale of 2.0.")\n')
            text = _replace(text, "paths = [str(p.GetPath()) for p in rig.links]\n",
                            'paths = [str(p.GetPath()) for p in rig.links]\n'
                            'times, samples = np.asarray(rig.times), np.asarray(rig.samples)\n'
                            'if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()\n'
                            '        or not np.all(np.diff(times) > 0) or abs(times[0]) > 1e-10 or times[-1] < 8.4\n'
                            '        or samples.shape != (len(times), len(paths), 7) or not np.isfinite(samples).all()\n'
                            '        or not paths or any(not path.startswith("/World/RobotPourDemo/Franka/") for path in paths)):\n'
                            '    raise RuntimeError("The owned robot kinematic bank has an invalid clock, shape, path or finite state.")\n')
            text = _replace(text, "out = Path(r'C:\\omninverse\\projects\\pour_policy_training_v3')",
                            'if not np.isfinite(matrices).all():\n'
                            '    raise RuntimeError("Nonfinite exported robot FK matrices.")\n'
                            'out = ROOT / "_cache/pour_training/fast_v3"\n'
                            'if (out / "legacy_kinematic_bank.npz").exists() or (out / "legacy_kinematic_bank.json").exists():\n'
                            '    raise FileExistsError("Preserve the existing robot bank; use a clean local preparation directory.")')
        else:
            raise SyncError("Unreviewed fast V3 tool: " + name)
    else:
        raise SyncError("Unreviewed fast V3 source: " + relative)
    return text


def transform(relative, text):
    """Return the fixed publication variant; never evaluate imported scripts."""
    if relative in FAST_RUNTIME_SOURCES or relative in FAST_TOOL_SOURCES:
        text = _transform_fast_source(relative, text)
    elif relative == "droplet_milkcrown_newton.py":
        text = _replace(text, "import os\n", "import os\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'WOOD_TEXTURE = r"C:\\omninverse\\textures\\wooden_table.jpg"',
                        'WOOD_TEXTURE = str(PROJECT_ROOT / "textures" / "wooden_table.jpg")')
        text = _replace(text, "/mnt/c/omninverse/", "./", count=4)
        text = _replace(text, "# Milk-crown cup in Newton 1.7 implicit MPM.",
                        "# Milk-crown cup using the Isaac Sim bundled Newton implicit MPM solver.")
    elif relative == "droplet_milkcrown_newton_surface.py":
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'TABLE_STYLE = "original_wood"', 'TABLE_STYLE = "white_matte"')
        text = _replace(text, 'TEXTURE_DIRECTORY = Path(r"C:\\omninverse\\textures\\table_options")',
                        'TEXTURE_DIRECTORY = PROJECT_ROOT / "textures" / "table_options"')
        text = _replace(text, '    r"C:\\omninverse\\droplet_milkcrown_newton.py",',
                        '    str(PROJECT_ROOT / "droplet_milkcrown_newton.py"),')
    elif relative == "droplet_milkcrown_newton_view.py":
        text = _replace(text, "import importlib.util\n",
                        "import importlib.util\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, '    r"C:\\omninverse\\droplet_milkcrown_newton_surface.py",',
                        '    str(PROJECT_ROOT / "droplet_milkcrown_newton_surface.py"),')
    elif relative in ("robot_testtube_pour.py", "robot_target_volume_pour.py"):
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'TABLE_STYLE = "pale_oak"', 'TABLE_STYLE = "white_matte"')
        text = _replace(text, 'INSTALL = Path(r"C:\\omninverse")', 'INSTALL = PROJECT_ROOT')
        if relative == "robot_testtube_pour.py":
            text = _replace(text, 'OUTPUT = Path(r"C:\\omninverse\\droplet_out\\robot_testtube_pour")',
                            'OUTPUT = PROJECT_ROOT / "droplet_out" / "robot_testtube_pour"')
            text = _replace(text, 'LIQUID_MODE = "replay"', 'LIQUID_MODE = "simulate"')
    elif relative == "robot_target_volume_ai.py":
        text = _replace(text, "from pathlib import Path\n",
                        "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'INSTALL = Path(os.environ.get("ISAAC_SIM_ROOT", r"C:\\omninverse"))',
                        'INSTALL = PROJECT_ROOT')
    elif relative == "robot_target_volume_precision.py":
        text = _replace(text, "from pathlib import Path\n",
                        "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'INSTALL = Path(os.environ.get("ISAAC_FLUID_PROJECT_ROOT", r"C:\\omninverse"))',
                        'INSTALL = PROJECT_ROOT')
    elif relative == "train_target_volume_policy.py":
        text = _replace(text, "from pathlib import Path\n",
                        "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT, require_isaac_sim_root\n")
        text = _replace(text, "ROOT = Path(__file__).resolve().parent", "ROOT = PROJECT_ROOT")
        text = _replace(text, '    sdk = Path(os.environ.get("ISAAC_SIM_ROOT", str(ROOT))).resolve()',
                        '    sdk = require_isaac_sim_root()')
        text = _replace(text, '    sdk = Path(os.environ.get("ISAAC_SIM_ROOT", str(ROOT)))',
                        '    sdk = require_isaac_sim_root()')
    elif relative == "train_target_volume_policy_full.py":
        text = _replace(text, "from pathlib import Path\n",
                        "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT, require_isaac_sim_root\n")
        text = _replace(text, "ROOT = Path(__file__).resolve().parent", "ROOT = PROJECT_ROOT")
        text = _replace(text, '    sdk = Path(os.environ.get("ISAAC_SIM_ROOT", str(ROOT)))',
                        '    sdk = require_isaac_sim_root()')
    elif relative == "train_target_volume_precision.py":
        text = _replace(text, "from pathlib import Path\n",
                        "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, "ROOT = Path(__file__).resolve().parent", "ROOT = PROJECT_ROOT")
        text = _replace(text, '    parser.add_argument("--configs", type=Path, required=True)',
                        '    parser.add_argument("--configs", type=Path, default=PROJECT_ROOT / "configs/precision_v2_calibration.json")')
        text = _replace(text,
                        '    parser.add_argument("--output", type=Path, default=ROOT / "droplet_out/pour_policy_training/run_20261122_precision")',
                        '    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated")')
    elif relative in CHECKPOINT_SOURCES:
        _validate_checkpoint(relative, text)
    elif relative == "projects/pour_policy_training/audit_candidate.py":
        _replace(text, '"""Supplementary live audit of the already-trained CEM candidate.',
                 '"""Supplementary live audit of the already-trained CEM candidate.')
        _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                 "ROOT = Path(__file__).resolve().parents[2]")
    elif relative == "projects/pour_policy_training/plot_results.py":
        _replace(text, '"""Plot only a completed pouring-training run, using its recorded results.',
                 '"""Plot only a completed pouring-training run, using its recorded results.')
        _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                 "ROOT = Path(__file__).resolve().parents[2]")
        text = _replace(text, r"C:\\omninverse\\kit\\python\\python.exe plot_results.py RUN_DIRECTORY",
                        "SDK_PYTHON tools/training/plot_results.py RUN_DIRECTORY")
    elif relative == "projects/pour_policy_training/plot_full_results.py":
        _replace(text, '"""Plot only a completed, internally consistent full-budget pouring run.',
                 '"""Plot only a completed, internally consistent full-budget pouring run.')
        _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                 "ROOT = Path(__file__).resolve().parents[2]")
    elif relative in ("projects/pour_policy_training_v2/audit_precision_run.py",
                      "projects/pour_policy_training_v2/plot_precision_results.py"):
        _replace(text, "ROOT = Path(__file__).resolve().parents[2]",
                 "ROOT = Path(__file__).resolve().parents[2]")
    elif relative == PRECISION_CONFIG_SOURCE:
        configurations = json.loads(text)
        if not isinstance(configurations, list) or len(configurations) != 3:
            raise SyncError("Precision calibration requires three predeclared configurations.")
        names = set()
        for configuration in configurations:
            if not isinstance(configuration, dict) or set(configuration) != {"name", "params"}:
                raise SyncError("Unexpected precision calibration configuration fields.")
            name = configuration["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name) or name in names:
                raise SyncError("Precision configuration names must be unique short identifiers.")
            names.add(name)
            _validate_precision_parameters(configuration["params"])
        json.dumps(configurations, allow_nan=False)
    elif relative.endswith("/scripts/robot.py"):
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT, ISAAC_SIM_ROOT\n")
        text = _replace(text,
                        "FRANKA_PACKAGE = Path(r'C:\\omninverse\\exts\\isaacsim.asset.importer.urdf\\data\\urdf\\robots\\franka_description')",
                        "FRANKA_PACKAGE = ISAAC_SIM_ROOT / 'exts' / 'isaacsim.asset.importer.urdf' / 'data' / 'urdf' / 'robots' / 'franka_description'")
        text = _replace(text, "FRANKA_CACHE = Path(r'C:\\omninverse\\assets\\robot_testtube_pour\\franka')",
                        "FRANKA_CACHE = PROJECT_ROOT / 'assets' / 'robot_testtube_pour' / 'franka'")
    elif relative.endswith("/scripts/video.py"):
        pattern = r"os\.environ\.get\('LOCALAPPDATA', r'C:\\Users\\[^\\/'\r\n]+\\AppData\\Local'\)"
        text, found = re.subn(pattern,
                             "os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))", text)
        if found != 1:
            raise SyncError(f"Expected exactly one LOCALAPPDATA Windows-user fallback; found {found}.")
    elif relative.endswith("/README.md"):
        text = _replace(text, r"C:\omninverse", "<PROJECT_ROOT>", count=5)
        text = _replace(text, "# 机器人用试管向烧杯倒水\n", "# 机器人用试管向烧杯倒水\n\n"
                        "> 公开仓库默认 `white_matte` 桌面与 `simulate` 实时求解；原实验的回放、视频与节奏说明"
                        "适用于可选物理缓存。Franka 从 SDK 本地转换，缓存不随源码提供。"
                        "`<PROJECT_ROOT>` 表示本仓库目录，启动配置见根目录 README。\n")
        text = _replace(text, "入口：`<PROJECT_ROOT>\\robot_testtube_pour.py`。",
                        "入口：项目根目录的 `robot_testtube_pour.py`。")
        text = _replace(text, "在 Isaac Sim 的 Window > Script Editor 中打开入口文件，点击 Run。",
                        "在 Isaac Sim 的 Window > Script Editor 中使用[根目录 README 的加载片段](../../README.md#环境与启动)，"
                        "保留其中的 `sys.path` 和真实 `__file__` 设置，并将 `script` 改为 `project / \"robot_testtube_pour.py\"`，再点击 Run。")
        text = _replace(text, "点击 Export 10s MP4 直接生成短片。",
                        "先生成可选物理缓存并切换到 `replay`，才可点击 Export 10s MP4 生成短片。")
        text = _replace(text,
                        "也可在 PowerShell 中从安装目录启动：\n\n```powershell\n"
                        "cd <PROJECT_ROOT>\n.\\isaac-sim.bat --exec <PROJECT_ROOT>\\robot_testtube_pour.py\n```",
                        "也可在 PowerShell 中设置独立的 SDK 安装目录，并从仓库根目录启动。把下方 SDK 路径改为实际安装位置：\n\n"
                        '```powershell\n$env:ISAAC_SIM_ROOT = "D:\\apps\\isaac-sim"\n'
                        '$env:ISAAC_FLUID_DEMO = "pour"\n'
                        '& "$env:ISAAC_SIM_ROOT\\isaac-sim.bat" --exec "$PWD\\launch.py"\n```')
        text = _replace(text,
                        "默认回放约 9.85 秒，扶正后结束。倒水后的等待为 0.25 秒，扶正用时 1.2 秒。",
                        "公开版默认实时求解，仿真时间约 15.6 秒，扶正后结束。可选缓存回放约 9.85 秒；"
                        "回放中的倒水后等待为 0.25 秒，扶正用时 1.2 秒。")
        text = _replace(text,
                        '- `LIQUID_MODE = "replay"`：默认，回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。\n'
                        '- `LIQUID_MODE = "simulate"`：重新实时求解液体，计算量更大。修改水粒子间距、运动时长或容器几何后使用此模式。',
                        '- `LIQUID_MODE = "simulate"`：公开版默认，实时求解液体，不需要物理缓存。修改水粒子间距、运动时长或容器几何后也使用此模式。\n'
                        '- `LIQUID_MODE = "replay"`：可选，先按[根目录 README](../../README.md#缓存桌面与视频)生成 `pour_cache.npz`，'
                        '再回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。缓存不随源码发布。')
        text = _replace(text, "默认推荐直接使用本示例的 Export 按钮，",
                        "在已准备好缓存的 `replay` 模式下推荐使用本示例的 Export 按钮，")
        text = _replace(text,
                        "`scripts/` 中四个模块分别提供几何、机器人、液体和视频导出实现。`franka/` 是官方 URDF 转换后的必要资源；"
                        "`pour_cache.npz` 是完整物理轨迹，不是视频。材质复用现有 milkcrown surface 脚本及桌面纹理。",
                        "`scripts/` 提供几何、机器人、液体、反馈控制与视频导出实现。`franka/` 是首次运行时从 SDK 自带 URDF 转换生成的本地缓存；"
                        "可选的 `pour_cache.npz` 是物理轨迹，不是视频，可通过 `tools/bake_pour_cache.py` 生成。两类缓存均不进入 Git。"
                        "材质复用 milkcrown surface 脚本，默认纯色桌面无需贴图。")
        text = _replace(text,
                        "两个文件可以同时在 Script Editor 中打开；选择目标文件并点击 Run 才会切换场景。"
                        "磁盘文件更新后，旧标签不会自动读取新代码，应关闭后重新打开该文件。",
                        "可以在 Script Editor 中保留多个加载片段；用根目录 README 的方式选择入口并点击 Run 切换场景。"
                        "该加载片段每次从磁盘读取源码，更新后重新运行片段即可。")
    elif relative.endswith("/TARGET_VOLUME.md"):
        text = _replace(text, r"C:\omninverse", "<PROJECT_ROOT>", count=2)
        text = _replace(text, "# 目标水量倒水\n", "# 目标水量倒水\n\n"
                        "> `<PROJECT_ROOT>` 表示本仓库目录。公开版默认使用无需下载贴图的 `white_matte` 桌面，"
                        "SDK 与项目路径配置见根目录 README。\n")
        text = _replace(text, "新入口：`<PROJECT_ROOT>\\robot_target_volume_pour.py`。",
                        "入口：项目根目录的 `robot_target_volume_pour.py`。")
        text = _replace(text,
                        "在 Isaac Sim 中选择 Window > Script Editor，在 File > Open 中打开新入口并 Run。",
                        "在 Isaac Sim 中选择 Window > Script Editor，使用[根目录 README 的加载片段](../../README.md#环境与启动)，"
                        "保留 `sys.path` 和真实 `__file__` 设置，将 `script` 指向 `project / \"robot_target_volume_pour.py\"` 后 Run。")
        text = _replace(text,
                        "文件顶部 `TARGET_VOLUME_ML = 30.0` 设置初始目标。",
                        "也可从仓库根目录在 PowerShell 中启动，把 SDK 路径改为实际安装位置：\n\n"
                        '```powershell\n$env:ISAAC_SIM_ROOT = "D:\\apps\\isaac-sim"\n'
                        '$env:ISAAC_FLUID_DEMO = "target-volume"\n'
                        '& "$env:ISAAC_SIM_ROOT\\isaac-sim.bat" --exec "$PWD\\launch.py"\n```\n\n'
                        "文件顶部 `TARGET_VOLUME_ML = 30.0` 设置初始目标。")
        text = _replace(text,
                        "原 `robot_testtube_pour.py` 的回放加速和 10 秒视频设置不应用于本例。",
                        "本例始终实时求解；旧 `robot_testtube_pour.py` 入口的公开默认也是 `LIQUID_MODE=\"simulate\"`，"
                        "其 `replay` 是需要另生成物理缓存的可选模式。旧入口的回放加速和 10 秒视频设置不应用于本例。")
    elif relative.endswith("/AI_TRAINING.md"):
        text = _replace(text,
                        "入口：`robot_target_volume_ai.py`。训练完成后，在 Isaac Sim 的 Script Editor 中打开并 Run，选择目标水量，再按 Play。",
                        "入口：项目根目录的 `robot_target_volume_ai.py`。结果与模型选择见 `results/ai_training.json`；"
                        "冻结检查点准备好后，在 Isaac Sim Script Editor 中使用[根目录 README 的加载片段](../../README.md#环境与启动)，"
                        "保留 `sys.path` 与真实 `__file__`，将 `script` 指向该入口，选择目标水量后按 Play。")
        text = _replace(text,
                        '& "C:\\omninverse\\kit\\python\\python.exe" "C:\\omninverse\\train_target_volume_policy.py" --quick --graph '
                        '--output "C:\\omninverse\\droplet_out\\pour_policy_training\\my_run"',
                        '$env:ISAAC_SIM_ROOT = "D:\\apps\\isaac-sim"\n'
                        '& "$env:ISAAC_SIM_ROOT\\kit\\python\\python.exe" "$PWD\\train_target_volume_policy.py" --quick --graph '
                        '--output "$PWD\\droplet_out\\pour_policy_training\\run_quick"')
        text = _replace(text, "从 PowerShell 运行一轮训练：",
                        "从仓库根目录的 PowerShell 运行一轮训练，并将 `ISAAC_SIM_ROOT` 改为实际 SDK 安装位置：")
    elif relative.endswith("/geometry.py"):
        _replace(text, '"""Pure NumPy geometry and prescribed motion for the robot pouring example.',
                 '"""Pure NumPy geometry and prescribed motion for the robot pouring example.')
    elif relative.endswith("/liquid.py"):
        _replace(text, '"""Fixed-particle GPU position-based liquid for the robot pouring example.',
                 '"""Fixed-particle GPU position-based liquid for the robot pouring example.')
    elif relative.endswith("/target_control.py"):
        _replace(text, '"""Feedback dosing controller; pure NumPy, no particle/state modifications.',
                 '"""Feedback dosing controller; pure NumPy, no particle/state modifications.')
    elif relative.endswith("/volume_visuals.py"):
        _replace(text, '"""Display-only water contrast and equivalent-volume guides for dosing.',
                 '"""Display-only water contrast and equivalent-volume guides for dosing.')
    elif relative.endswith("/pour_training_env.py"):
        _replace(text, '"""Live PBF environment for goal-conditioned pouring policy experiments.',
                 '"""Live PBF environment for goal-conditioned pouring policy experiments.')
    elif relative.endswith("/learned_policy.py"):
        _replace(text, '"""Small, interpretable feedback policy trained by episodic CEM search.',
                 '"""Small, interpretable feedback policy trained by episodic CEM search.')
    elif relative.endswith("/pbf_graph.py"):
        _replace(text, '"""Optional CUDA graph dispatch for the existing PourFluid PBF substep.',
                 '"""Optional CUDA graph dispatch for the existing PourFluid PBF substep.')
    elif relative.endswith("/precision_control.py"):
        _replace(text, '"""Experimental closed-loop particle-pulse dosing; no liquid-state mutations.',
                 '"""Experimental closed-loop particle-pulse dosing; no liquid-state mutations.')
    elif relative.endswith("/precision_training_env.py"):
        _replace(text, '"""Provenance and complete count logging for live precision-pouring trials.',
                 '"""Provenance and complete count logging for live precision-pouring trials.')
    elif relative == "textures/table_options/sources.json":
        manifest = json.loads(text)
        if set(manifest) != {"gray_concrete", "pale_oak", "white_marble"}:
            raise SyncError("The CC0 manifest's asset allowlist changed; review the sync mapping.")
        if any(item.get("license") != "CC0 1.0" for item in manifest.values()):
            raise SyncError("Only the reviewed CC0 texture manifest can be copied.")
    else:
        raise SyncError(f"No publication mapping for {relative}.")
    if "C:\\omninverse" in text or re.search(r"(?i)[a-z]:\\Users\\", text):
        raise SyncError(f"Unmapped machine-specific path or username in {relative}.")
    if relative.endswith(".py"):
        ast.parse(text, filename=relative)
    return text if relative in CHECKPOINT_SOURCES else text.rstrip() + "\n"


def _fast_publication_gate(root):
    """Completed failed research is publishable; it is never a GUI default."""
    run = root / FAST_RUN
    required = ("summary.json", "protocol.json", "input_freeze.json", "freeze.json", "episodes.jsonl", "fast_run_audit.json")
    if any(not (run / name).is_file() for name in required):
        raise SyncError("Fast V3 publication requires complete summary/freeze/journal and independent raw audit.")
    summary, envelope, freeze, audit = (json.loads((run / name).read_text(encoding="utf-8")) for name in
                                        ("summary.json", "protocol.json", "freeze.json", "fast_run_audit.json"))
    protocol = envelope["protocol"]
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    budget = {"generations": 3, "population": 6, "elites": 2, "training": 108, "validation": 12, "test": 18, "total": 138}
    if (summary.get("complete") is not True or summary.get("total_recorded_episodes") != 138
            or protocol.get("budget") != budget or summary.get("protocol") != protocol
            or identity != envelope.get("protocol_id") or summary.get("protocol_id") != identity
            or freeze.get("protocol_id") != identity or audit.get("protocol_id") != identity
            or audit.get("artifact_integrity_verified") is not True or audit.get("all_failures_retained") is not True
            or audit.get("episodes_audited") != 138 or audit.get("raw_action_trace_count") != 138
            or audit.get("summary_sha256") != hashlib.sha256((run / "summary.json").read_bytes()).hexdigest()
            or audit.get("journal_sha256") != hashlib.sha256((run / "episodes.jsonl").read_bytes()).hexdigest()
            or summary.get("automatic_gui_deployment") is not False or freeze.get("test_used_for_selection") is not False
            or summary.get("test_used_for_training_or_selection") is not False):
        raise SyncError("Fast V3 requires the complete, unchanged, independently audited 108/12/18 experiment.")
    sources = protocol.get("source_sha256", {})
    if len(sources) != 20:
        raise SyncError("Fast V3 must bind exactly twenty frozen sources.")
    for relative, expected in sources.items():
        source = (root / relative).resolve()
        frozen_source = (run / "source_snapshot" / relative).resolve()
        if (relative not in ALLOWLIST or not source.is_relative_to(root) or not frozen_source.is_relative_to(run)
                or not source.is_file() or not frozen_source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected
                or hashlib.sha256(frozen_source.read_bytes()).hexdigest() != expected):
            raise SyncError("Fast V3 source changed or is not reviewed: " + relative)
    lines = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(lines) != 138 or len({row["episode_id"] for row in lines}) != 138
            or {stage: sum(row["conditions"]["stage"] == stage for row in lines) for stage in ("train", "validation", "test")}
            != {"train": 108, "validation": 12, "test": 18}
            or any(datetime.fromisoformat(row["episode_wall_clock_start_utc"]) <= datetime.fromisoformat(freeze["frozen_at_utc"])
                   for row in lines if row["conditions"]["stage"] == "test")):
        raise SyncError("Fast V3 journal budgets or pre-test model freeze differ.")
    for role in ("candidate", "recommended"):
        path = run / (role + "_frozen.json")
        raw = path.read_bytes()
        _validate_fast_checkpoint(raw.decode("utf-8"), role)
        model, entry = json.loads(raw), summary["models"][role]
        if (hashlib.sha256(raw).hexdigest() != entry["sha256"] or entry["sha256"] != freeze[role + "_sha256"]
                or model["parameters"] != entry["parameters"] or model["parameters"] != freeze[role + "_params"]
                or model["metadata"]["source_sha256"] != sources or model["metadata"]["protocol_id"] != identity):
            raise SyncError("Fast V3 frozen model SHA/parameters/metadata changed.")
    return summary


def prepare_sync(workspace):
    """Validate the entire allowlist before returning any content to write."""
    root = Path(workspace).expanduser().resolve()
    prepared = []
    precision_source_manifest = None
    fast_summary = None
    for relative in ALLOWLIST:
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise SyncError(f"Missing or unsafe allowlisted source: {relative}.")
        # Model bytes retain their original line endings so the published
        # checkpoint has exactly the same SHA as the frozen local artifact.
        original = (source.read_bytes().decode("utf-8") if relative in CHECKPOINT_SOURCES
                    else source.read_text(encoding="utf-8"))
        try:
            if relative in FAST_MODEL_SOURCES:
                if fast_summary is None:
                    fast_summary = _fast_publication_gate(root)
                role = FAST_MODEL_SOURCES[relative][1]
                entry = fast_summary["models"][role]
                if hashlib.sha256(original.encode("utf-8")).hexdigest() != entry["sha256"]:
                    raise SyncError("Fast checkpoint changed during publication preparation.")
            if relative in PRECISION_MODEL_SOURCES:
                summary_path = root / PRECISION_RUN / "summary.json"
                if not summary_path.is_file():
                    raise SyncError("Completed precision summary is missing; publication is premature.")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                checkpoint = json.loads(original)
                role = PRECISION_MODEL_SOURCES[relative][1]
                entry = summary.get("models", {}).get(role, {})
                protocol = summary.get("protocol", {})
                if (summary.get("complete") is not True
                        or type(summary.get("total_recorded_episodes")) is not int
                        or summary["total_recorded_episodes"] != 48
                        or [protocol.get(key) for key in ("calibration_episodes", "validation_episodes", "test_episodes", "total_live_episodes")]
                        != [18, 12, 18, 48]):
                    raise SyncError("Precision checkpoint requires the completed 48-episode protocol.")
                if (not summary.get("protocol_id")
                        or summary["protocol_id"] != checkpoint.get("metadata", {}).get("protocol_id")
                        or hashlib.sha256(original.encode("utf-8")).hexdigest() != entry.get("sha256")
                        or checkpoint.get("parameters") != entry.get("parameters")
                        or checkpoint.get("metadata", {}).get("source_sha256") != protocol.get("source_sha256")):
                    raise SyncError("Precision checkpoint does not match completed protocol/model SHA/parameters.")
                expected_sources = protocol.get("source_sha256")
                if precision_source_manifest is None:
                    if not isinstance(expected_sources, dict) or len(expected_sources) != 11:
                        raise SyncError("Precision protocol must identify its eleven frozen sources.")
                    for source_name, expected_sha in expected_sources.items():
                        if not isinstance(source_name, str) or Path(source_name).is_absolute():
                            raise SyncError("Unsafe precision source path in completed protocol.")
                        actual_source = (root / source_name).resolve()
                        if (not actual_source.is_relative_to(root) or not actual_source.is_file()
                                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha))
                                or hashlib.sha256(actual_source.read_bytes()).hexdigest() != expected_sha):
                            raise SyncError("Current workspace precision source differs from the frozen run: " + source_name)
                    precision_source_manifest = dict(expected_sources)
                elif expected_sources != precision_source_manifest:
                    raise SyncError("Precision model source manifests changed during sync preparation.")
                if relative == "assets/robot_testtube_pour/policy/precision_v2.json":
                    if summary.get("release_acceptance", {}).get("recommended_passed") is not True:
                        raise SyncError("GUI precision default requires the frozen recommendation to pass release acceptance.")
                    gui_path = summary_path.parent / "gui_validation/validation.json"
                    if not gui_path.is_file():
                        raise SyncError("GUI precision default requires completed three-target visible validation.")
                    gui = json.loads(gui_path.read_text(encoding="utf-8"))
                    results = gui.get("results", [])
                    if (gui.get("complete") is not True or type(gui.get("additional_gui_episodes")) is not int
                            or gui["additional_gui_episodes"] != 3
                            or gui.get("physical_integration_passed") is not True
                            or gui.get("dose_within_0_1_ml") is not True
                            or gui.get("checkpoint_sha256") != entry.get("sha256")
                            or gui.get("protocol_id") != summary["protocol_id"]
                            or gui.get("cuda_graph") is not False
                            or gui.get("training_or_heldout_test") is not False
                            or not isinstance(results, list) or len(results) != 3
                            or any(not isinstance(result, dict) for result in results)
                            or {result.get("target_ml") for result in results} != {20, 30, 40}):
                        raise SyncError("GUI precision default lacks complete matching 20/30/40 mL acceptance.")
                    for result in results:
                        error = result.get("error_ml")
                        if (any(result.get(flag) is not True for flag in
                                ("success", "complete", "mass_conserved", "physical_integration_passed"))
                                or result.get("controller", {}).get("done") is not True
                                or result.get("precision_checkpoint_sha256") != entry["sha256"]
                                or type(error) not in (int, float) or not math.isfinite(error) or abs(error) > .1):
                            raise SyncError("A visible precision target failed its original 0.1 mL/physics/model checks.")
            if relative in FULL_MODEL_SOURCES:
                # These checkpoints are frozen before test. Publish them only
                # after the isolated full run has actually completed its budget.
                summary_path = source.parent / "summary.json"
                if not summary_path.is_file():
                    raise SyncError("Full-run summary is missing; checkpoint publication is premature.")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                checkpoint = json.loads(original)
                role = FULL_MODEL_SOURCES[relative][1]
                entry = summary.get("models", {}).get(role, {})
                if (summary.get("complete") is not True
                        or type(summary.get("total_recorded_episodes")) is not int
                        or summary["total_recorded_episodes"] != 90):
                    raise SyncError("Full-run checkpoint requires a completed 90-episode summary.")
                if (not summary.get("protocol_id")
                        or summary["protocol_id"] != checkpoint.get("metadata", {}).get("protocol_id")
                        or hashlib.sha256(original.encode("utf-8")).hexdigest() != entry.get("sha256")
                        or checkpoint.get("parameters") != entry.get("parameters")):
                    raise SyncError("Full-run checkpoint does not match the completed summary.")
            published = transform(relative, original)
        except (ValueError, SyntaxError, SyncError) as error:
            raise SyncError(f"{relative}: {error}") from error
        destination_relative = SOURCE_TO_DEST.get(relative, relative)
        prepared.append((destination_relative, published.encode("utf-8")))
    return prepared


def _write_atomic(destination, content):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true", help="Validate all anchors and list changes; write nothing.")
    args = parser.parse_args(argv)
    prepared = prepare_sync(args.workspace)
    changed = []
    for relative, content in prepared:
        destination = PROJECT_ROOT / relative
        if not destination.resolve().is_relative_to(PROJECT_ROOT):
            raise SyncError(f"Unsafe publication destination: {relative}.")
        if not destination.is_file() or destination.read_bytes() != content:
            changed.append((relative, content))
    if not args.dry_run:
        for relative, content in changed:
            _write_atomic(PROJECT_ROOT / relative, content)
    print(json.dumps({"dry_run": args.dry_run, "validated_files": len(prepared),
                      "changed_files": [name for name, _ in changed],
                      "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in prepared}},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SyncError, ValueError) as error:
        print(f"Sync aborted: {error}", file=sys.stderr)
        raise SystemExit(1)
