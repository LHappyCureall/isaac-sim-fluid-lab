"""Explicit V12 research publication; no SDK, raw trajectories or GUI default.

The existing121 aliases stay unchanged. This independent bundle selects only
the completed collection's effective source snapshots, the separately frozen
goal-evaluation additions, reviewed post-hoc tools and two small model JSONs.
"""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


RUN = "droplet_out/pour_policy_adaptive_response/collection_001"
GOAL = "droplet_out/pour_policy_adaptive_response/goal_development_001"
FIT = "droplet_out/pour_policy_adaptive_response/fit_001"
HERE = "projects/pour_policy_adaptive_learning/"
PUBLIC = "research/adaptive_response_v12_20261127/"
REPORT = "results/adaptive_response_v12_20261127.json"
PLOTS = "droplet_out/pour_policy_adaptive_response/plots_002"
PLOT_PUBLIC = "results/adaptive_response_v12_20261127/"
PLOT_NAMES = ("response_curves.png", "validation_response_fit.png", "goal_policy_evaluation.png")
PREFIX_EXTENSION_NAMES = tuple("projects/pour_policy_adaptive_learning/" + name for name in
    ("prefix_control.py", "run_prefix_calibration.py", "audit_prefix_calibration.py",
     "test_prefix_control.py", "test_audit_prefix_calibration.py", "curate_prefix_results.py"))
MODEL_KIND = "empirical_anchor_approach_micro_cutoff_cycle_v1"
PREFIX = "assets/robot_testtube_pour/scripts/"
RUNTIME_NAMES = ("train_target_volume_policy.py", *[PREFIX + name for name in (
    "geometry.py", "liquid.py", "target_control.py", "robot.py", "pour_training_env.py",
    "fast_geometry.py", "fast_control.py", "fast_training_env.py", "direct_fast_geometry.py",
    "direct_fast_control.py", "direct_training_env.py", "mouth_pivot_fast_geometry.py",
    "mouth_pivot_lift_geometry.py", "mouth_pivot_lift_control_v2.py")],
    *["projects/pour_policy_training_v9/" + name for name in
      ("legacy_control.py", "fine_control.py", "fine_env.py", "run_fine_search.py")],
    "projects/pour_policy_training_v10/range_control.py",
    *[HERE + name for name in ("response_control.py", "cycle_control.py", "response_samples.py",
                              "response_env.py", "cycle_env.py", "collect_cycles.py")])
EXTRA_RUNTIME = {
    HERE + "evaluate_cycle_policy.py": "790d634555a216e0dda8dbdf50b69f8d425a43e4e3c5e3ac73c81fc217c491b0",
    HERE + "cycle_response_model.py": "7828032a41284123ed89a3340bb72d5b790c12e4e49ebd6ee7ae6989a6c723e6",
}
TOOLS = {
    HERE + "audit_response_collection.py": "3fc83116776a204349dcf76242052e8f40749b485efa3513aebdf6cdaaadc7d9",
    HERE + "fit_cycle_model_cli.py": "61bb72ee5ed7e8947f1abfbbdd9536d0c7f194cd3b337b95f59f21ee483377fc",
    HERE + "cycle_samples.py": "cfd4838eacbc786f6ac1fdb27a48bfbe7e377cceaed66136a7ec17e53cf88468",
    HERE + "curate_response_results.py": "0c0955cfa7f43d76bf807de4a6043d4ab8b9d8f04ca4f3b2d1994a84de5bfd45",
    HERE + "plot_response_results.py": "2233b36ba4c17d4e37ac44a8e35aebf1a6d9ba16358736327c81ccdab4ca4e95",
    "projects/pour_policy_training_v9/audit_fine_run.py": "cadf9e35089db98a1d971e1476229e856342a802628f95c414ceb244f27f5c83",
}
ROBOT = PREFIX + "robot.py"
ROBOT_ORIGINAL = "6b735760b81ccfb6733db620cbb852e1fabd55cb5b0fed662d03fe17095e6a1d"
ROBOT_PUBLIC = "972bab813c3caa0f8a1d68e96652594de36b77644183bb737cb86d23877975b3"
ATTRIBUTES = ("# Preserve frozen source and small model bytes.\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n"
    "*.py -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "models/*.json -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n")
README = ("# V12 real-response research bundle\n\n"
    "This is supervised identification of measured anchor-to-anchor liquid responses and a frozen goal-conditioned "
    "controller experiment, not PPO and not a default GUI upgrade. Consult "
    "[the actual results](../../docs/ADAPTIVE_RESPONSE_V12_20261127.md).\n\n"
    "The26 collection runtime sources come from the completed collection's source snapshots. "
    "The evaluation factory and learned-model implementation were frozen separately for the subsequent goal experiment; "
    "six post-hoc tools/dependencies are distinct from the runtime. source_manifest.json records roles and every "
    "original/published SHA. Only robot.py converts two machine-specific resource locations to ISAAC_SIM_ROOT and this "
    "research-tree root. All other source bytes and both model JSONs remain unchanged.\n\n"
    "The model was fitted only on declared training responses; the validation split selected its configuration. "
    "Its local empirical uncertainty and abstention are not a certified dosing guarantee. Real settled measurements "
    "still decide dose, full-home completion, deadline, peak all-label spill and late-water acceptance. "
    "The source solver remains livePBF with privileged particle-mass truth and kinematic pregrasp.\n\n"
    "No SDK, robot mesh/URDF, particle or robot-path NPZ, raw step log, reset bank or video is distributed. "
    "Prepare licensed robot resources under your ISAAC_SIM_ROOT and rebuild the required local geometric path inputs "
    "using the separately documented preparation tools. Geometry caches are robot trajectories, not fluid replay. "
    "Rebuilt inputs and the portable robot source are new versions: retain literal guards, use a fresh protocol and "
    "engineering validation, and never rebind hashes or resume the private original journals. "
    "This provenance-preserving bundle is not an out-of-box replay or a validated real-robot controller.\n")


class PublicationError(ValueError):
    pass


def need(value, message):
    if not value:
        raise PublicationError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def inside(root, name):
    path = (root / name).resolve()
    need(path.is_relative_to(root.resolve()), "Publication path escaped its named root")
    return path


def private_paths(data):
    # SDK source text may discuss paths, but no user-specific literal location
    # is permitted in this public derivative or any selected JSON metadata.
    need(not re.search(rb"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\"'\s,\)\]]+|file://[^\"'\s]+|(?<![A-Za-z0-9_])/(?:Users|home)/[^\"'\s]+|AppData[\\/]", data),
         "Private absolute location in adaptive response publication")


def robot_bytes(name, raw):
    if name != ROBOT:
        return raw
    need(digest(raw) == ROBOT_ORIGINAL, "Unexpected robot source for two-path conversion")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf8")
    pairs = (("from pathlib import Path" + newline, "from pathlib import Path" + newline + "import os" + newline),
        ("FRANKA_PACKAGE = Path(r'C:\\omninverse\\exts\\isaacsim.asset.importer.urdf\\data\\urdf\\robots\\franka_description')",
         "FRANKA_PACKAGE = Path(os.environ['ISAAC_SIM_ROOT']) / 'exts' / 'isaacsim.asset.importer.urdf' / 'data' / 'urdf' / 'robots' / 'franka_description'"),
        ("FRANKA_CACHE = Path(r'C:\\omninverse\\assets\\robot_testtube_pour\\franka')",
         "FRANKA_CACHE = Path(__file__).resolve().parents[3] / 'assets' / 'robot_testtube_pour' / 'franka'"))
    for old, new in pairs:
        need(text.count(old) == 1, "Portable robot conversion anchor changed")
        text = text.replace(old, new)
    result = text.encode("utf8")
    need(digest(result) == ROBOT_PUBLIC, "Portable robot derivative differs from reviewed bytes")
    return result


def load_curator(workspace):
    directory = workspace / HERE
    for name, expected in TOOLS.items():
        need(digest(inside(workspace, name).read_bytes()) == expected, "Post-hoc tool changed: " + name)
    need(digest((directory / "cycle_response_model.py").read_bytes()) == EXTRA_RUNTIME[HERE + "cycle_response_model.py"],
         "Fitted model implementation changed")
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("publication_checked_response_curator", directory / "curate_response_results.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("cycle_response_model", "fit_cycle_model_cli", "cycle_samples"):
            need(Path(sys.modules[name].__file__).resolve() == (directory / (name + ".py")).resolve(),
                 "Unexpected cached publication helper module")
    finally:
        sys.path.pop(0)
    return module


def publication_gate(workspace, project):
    # Completion is checked before decoding journals/model data. A failed
    # but complete study may be published honestly; unfinished work may not.
    for relative, count in ((RUN, 60), (GOAL, 10)):
        need((workspace / relative / "summary.json").is_file(),
             "Adaptive study is still incomplete; no publication data decoded")
        summary, job = read(workspace / relative / "summary.json"), read(workspace / relative / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and job.get("state") == "complete" and job.get("pid") is None,
             "Adaptive response publication requires completed60collection/10goal cases")
    selected = read(project / REPORT)
    private_paths((project / REPORT).read_bytes())
    collection_audit = workspace / RUN / "collection_audit.json"
    goal_audit = workspace / GOAL / "goal_audit.json"
    curator = load_curator(workspace)
    study = curator.load_study(workspace / RUN, collection_audit, digest(collection_audit.read_bytes()), workspace / FIT)
    goal = curator.load_goal_evaluation(workspace / GOAL, goal_audit, digest(goal_audit.read_bytes()), study)
    expected = curator.build_public_summary(study, goal)
    ignored = {"record_created_at_utc", "plots", "publication"}
    need({k: v for k, v in selected.items() if k not in ignored}
         == {k: v for k, v in expected.items() if k not in ignored},
         "Adaptive selected report differs from independently audited raw study")
    need(set(study["protocol"]["source_sha256"]) == set(RUNTIME_NAMES),
         "Adaptive runtime must retain the exact26 source roles")
    goal_plan = read(workspace / GOAL / "protocol.json")
    need(set(goal_plan["source_sha256"]) == set(RUNTIME_NAMES) | set(EXTRA_RUNTIME),
         "Goal runtime must retain original26 plus exacttwo new roles")
    for name, expected_sha in EXTRA_RUNTIME.items():
        need(goal_plan["source_sha256"][name] == expected_sha, "Goal runtime addition source changed")
    need(selected["terminal_goal_policy_acceptance_proven"] is False
         and selected["default_gui_deployment"] is False
         and selected["all_continuous_goals_guaranteed"] is False,
         "This failed research round cannot become a default/continuous-goal acceptance claim")
    return study, selected, goal_plan


def prepare(workspace, project):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    study, selected, goal_plan = publication_gate(workspace, project)
    files, mapping = [], []

    def add(source, destination, role, original_file, expected_sha):
        raw = inside(workspace, source).read_bytes()
        need(digest(raw) == expected_sha, "Explicit adaptive source bytes changed: " + source)
        data = robot_bytes(original_file, raw)
        private_paths(data)
        if destination.endswith(".py"):
            ast.parse(data.decode("utf8"), filename=destination)
        files.append((destination, data))
        mapping.append({"role": role, "original_file": original_file,
            "published_file": destination, "original_sha256": expected_sha,
            "published_sha256": digest(data),
            "transform": "two_robot_resource_paths" if original_file == ROBOT else "unchanged_raw_bytes"})

    for name in RUNTIME_NAMES:
        add(RUN + "/source_snapshot/" + name, PUBLIC + name, "collection_runtime", name,
            study["protocol"]["source_sha256"][name])
    for name, expected_sha in EXTRA_RUNTIME.items():
        add(GOAL + "/source_snapshot/" + name, PUBLIC + name, "goal_runtime_addition", name, expected_sha)
    for name, expected_sha in TOOLS.items():
        add(name, PUBLIC + name, "posthoc_tool_or_cpu_dependency", name, expected_sha)
    fit_manifest = read(workspace / FIT / "fit_manifest.json")
    for name, expected_sha in (("candidate.json", fit_manifest["candidate_raw_sha256"]),
                               ("fit_manifest.json", selected["response_model"]["fit_manifest_raw_sha256"])):
        add(FIT + "/" + name, PUBLIC + "models/" + name, "frozen_research_model" if name == "candidate.json"
            else "frozen_fit_metadata", FIT + "/" + name, expected_sha)
    plot_manifest = read(workspace / PLOTS / "plot_provenance.json")
    need(plot_manifest["complete"] is True and plot_manifest["synthetic_fixture"] is False
         and plot_manifest["protocol_id"] == selected["protocol_id"]
         and plot_manifest["plot_source_sha256"] == TOOLS[HERE + "plot_response_results.py"]
         and plot_manifest["curator_source_sha256"] == TOOLS[HERE + "curate_response_results.py"]
         and set(plot_manifest["files_sha256"]) == set(PLOT_NAMES), "Final real plot provenance differs")
    plots = []
    for name in PLOT_NAMES:
        raw = (workspace / PLOTS / name).read_bytes()
        need(raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == plot_manifest["files_sha256"][name],
             "Real selected plot changed: " + name)
        files.append((PLOT_PUBLIC + name, raw))
        plots.append({"file": PLOT_PUBLIC + name, "sha256": digest(raw)})
    manifest = {"schema_version": 1, "kind": "frozen_adaptive_response_research_publication_v12",
        "collection_protocol_id": selected["protocol_id"], "goal_protocol_id": goal_plan["protocol_id"],
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_collection_audit_sha256": selected["independent_audit_raw_sha256"],
        "independent_goal_audit_sha256": selected["goal_evaluation"]["audit_raw_sha256"],
        "collection_cases": 60, "goal_development_cases": 10, "default_gui_deployment": False,
        "physics_input_sha256": study["protocol"]["input_sha256"],
        "runtime_versions": read(workspace / RUN / "versions.json"),
        "full_range_acceptance_proven": False, "files": mapping, "plots": plots,
        "source_path_derivative_requires_new_protocol": True,
        "raw_logs_or_cache_assets_distributed": False}
    metadata = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    private_paths(metadata)
    files.extend(((PUBLIC + "source_manifest.json", metadata), (PUBLIC + "README.md", README.encode()),
                  (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len({name for name, _ in files}) == len(files), "Duplicate explicit publication destination")
    return files


def verify_public(project):
    """SDK-independent release gate for only the explicit public bundle."""
    project = Path(project).resolve()
    manifest = read(project / PUBLIC / "source_manifest.json")
    selected = read(project / REPORT)
    need(manifest["schema_version"] == 1 and manifest["kind"] == "frozen_adaptive_response_research_publication_v12"
         and manifest["collection_protocol_id"] == selected["protocol_id"]
         and manifest["report_file"] == REPORT
         and manifest["report_sha256"] == digest((project / REPORT).read_bytes()), "Public adaptive report binding differs")
    need(selected["complete"] is True and selected["kind"] == "audited_real_whole_cycle_response_identification_v12"
         and selected["task_contract"] == {"deadline_physical_s": 24., "relative_dose_tolerance": .001,
             "max_all_label_peak_spill_particles": 1, "full_home_quiet_frames": 12,
             "post_observation_physical_s": 2., "post_can_rescue_deadline": False},
         "Public adaptive task contract changed")
    need(selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and selected["terminal_goal_policy_acceptance_proven"] is False
         and manifest["default_gui_deployment"] is False and manifest["full_range_acceptance_proven"] is False,
         "A failed identification experiment was promoted to default/range acceptance")
    entries = manifest["files"]
    roles = {role: [entry for entry in entries if entry["role"] == role]
             for role in ("collection_runtime", "goal_runtime_addition", "posthoc_tool_or_cpu_dependency",
                          "frozen_research_model", "frozen_fit_metadata")}
    need(len(entries) == 36 and len({entry["published_file"] for entry in entries}) == 36,
         "Exact adaptive bundle file count/identity differs")
    need({entry["original_file"] for entry in roles["collection_runtime"]} == set(RUNTIME_NAMES)
         and {entry["original_file"] for entry in roles["goal_runtime_addition"]} == set(EXTRA_RUNTIME)
         and {entry["original_file"] for entry in roles["posthoc_tool_or_cpu_dependency"]} == set(TOOLS),
         "Adaptive publication source-role allowlist differs")
    need(len(roles["frozen_research_model"]) == len(roles["frozen_fit_metadata"]) == 1,
         "Adaptive research model/fit metadata roles differ")
    need(set(selected["source_sha256"]) == set(RUNTIME_NAMES), "Selected runtime role count differs")
    for entry in entries:
        path = inside(project, entry["published_file"])
        raw = path.read_bytes()
        need(digest(raw) == entry["published_sha256"], "Adaptive published file changed: " + entry["published_file"])
        private_paths(raw)
        name, original = entry["original_file"], entry["original_sha256"]
        if entry["role"] == "collection_runtime":
            need(original == selected["source_sha256"][name] and entry["published_file"] == PUBLIC + name,
                 "Collection source differs from selected frozen provenance")
        if name == ROBOT:
            need(original == ROBOT_ORIGINAL and entry["published_sha256"] == ROBOT_PUBLIC
                 and entry["transform"] == "two_robot_resource_paths", "Robot public derivative changed")
        else:
            need(original == entry["published_sha256"] and entry["transform"] == "unchanged_raw_bytes",
                 "Unexpected public derivative or source rebinding")
        if entry["role"] in ("goal_runtime_addition", "posthoc_tool_or_cpu_dependency"):
            need(original == {**EXTRA_RUNTIME, **TOOLS}[name] and entry["published_file"] == PUBLIC + name,
                 "Reviewed goal/post-hoc implementation changed")
        if path.suffix == ".py":
            compile(raw.decode("utf8"), str(path), "exec")
    need((project / PUBLIC / "README.md").read_bytes() == README.encode()
         and (project / PUBLIC / ".gitattributes").read_bytes() == ATTRIBUTES.encode(), "Adaptive bundle usage/byte policy changed")
    expected = {entry["published_file"] for entry in entries} | {PUBLIC + name for name in
               ("source_manifest.json", "README.md", ".gitattributes")}
    if (project / PUBLIC / "prefix_source_manifest.json").is_file():
        # An explicitly versioned additive study has its own independent
        # manifest and release check. The original V12 manifest bytes stay put.
        expected.update(PUBLIC + name for name in PREFIX_EXTENSION_NAMES)
        expected.add(PUBLIC + "prefix_source_manifest.json")
    actual = {path.relative_to(project).as_posix() for path in (project / PUBLIC).rglob("*")
              if path.is_file() and not (path.suffix == ".pyc" and path.parent.name == "__pycache__")}
    need(actual == expected, "Unallowlisted SDK/cache/log/default file in adaptive research bundle")
    module_file = project / PUBLIC / HERE / "cycle_response_model.py"
    spec = importlib.util.spec_from_file_location("release_sealed_cycle_response_model", module_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    checkpoint = project / PUBLIC / "models/candidate.json"
    model = model_module.load_response_model(checkpoint, selected["response_model"]["candidate_raw_sha256"])
    need(model["kind"] == MODEL_KIND and model["schema_version"] == 1
         and model["source_sha256"] == EXTRA_RUNTIME[HERE + "cycle_response_model.py"]
         and model["selection_kind"] == "validation_only"
         and model["content_sha256"] == selected["response_model"]["model_content_sha256"],
         "Small empirical response model kind/schema/source changed")
    response = selected["response_model"]
    need(response["training_rows"] == model["training_row_count"] == 61
         and response["validation_rows"] == model["validation_row_count"] == 18
         and response["training_seeds"] == model["training_seeds"] == [190201, 190202, 190203]
         and response["validation_seeds"] == model["validation_seeds"] == [190301],
         "Empirical response model split/provenance count changed")
    validation = model["configuration_selection"][model["selected_config_index"]]["validation"]
    need(response["validation_predicted_rows"] == validation["predicted_rows"]
         and response["validation_abstained_rows"] == validation["abstained_rows"]
         and response["validation_mae_ml_on_supported_rows"] == validation["mean_absolute_delta_error_ml"]
         and response["validation_worst_ml_on_supported_rows"] == validation["maximum_absolute_delta_error_ml"]
         and response["validation_estimates_are_before_residual_envelope_calibration"] is True,
         "Response prediction metrics hid abstentions or reused post-calibrated validation")
    raw_fit = (project / PUBLIC / "models/fit_manifest.json").read_bytes()
    fit = json.loads(raw_fit)
    need(digest(raw_fit) == selected["response_model"]["fit_manifest_raw_sha256"]
         and fit["candidate_raw_sha256"] == digest(checkpoint.read_bytes())
         and fit["complete"] is True and fit["test_data_used"] is False and fit["dataset_splits_reassigned"] is False
         and fit["terminal_task_acceptance_proven"] is False and fit["default_gui_deployment"] is False,
         "Frozen fitting metadata/default role changed")
    physics = hashlib.sha256(json.dumps({"sources": selected["source_sha256"],
        "inputs": manifest["physics_input_sha256"], "spacing": .0027}, sort_keys=True).encode()).hexdigest()
    need(model["physics_fingerprint"] == physics and len(manifest["physics_input_sha256"]) == 6,
         "Original model physics provenance was rebound")
    cases = selected["cases"]
    need(len(cases) == manifest["collection_cases"] == 60 and len({row["episode_id"] for row in cases}) == 60,
         "Completed collection cases changed")
    need(sum(row["split"] == "train" for row in cases) == 45
         and sum(row["split"] == "validation" for row in cases) == 15, "Collection split count changed")
    for row in (*cases, *selected["goal_evaluation"]["cases"]):
        target, received = row["target_ml"], row["received_ml"]
        dose = target - .001 * target <= received <= target + .001 * target
        need(dose is row["dose_passed"] and abs(row["signed_error_ml"] - (received - target)) < 1e-9
             and abs(row["absolute_error_ml"] - abs(received - target)) < 1e-9,
             "Selected strict original-target dose gate changed")
        need(row["spill_passed"] is (row["peak_all_label_spill_particles"] <= 1)
             and (not row["full_home_within_deadline"] or row["physical_action_seconds"] <= 24.),
             "Selected spill/deadline gate changed")
        joint = bool(row["valid_evidence"] and dose and row["full_home_within_deadline"]
                     and row["spill_passed"] and row["no_late_dose_change"])
        need(row["strict_task_feasible"] is joint, "Selected joint task outcome changed")
    goal = selected["goal_evaluation"]
    need(len(goal["cases"]) == manifest["goal_development_cases"] == 10
         and all(row["split"] == "development" for row in goal["cases"])
         and goal["coverage"]["full_predeclared_range_suite"] is False
         and manifest["goal_protocol_id"] == goal["protocol_id"], "Development goals were promoted to a held-out range suite")
    for rows, metrics in ((cases, selected["collection_metrics"]), (goal["cases"], goal["metrics"])):
        need(metrics["cases"] == len(rows)
             and metrics["joint_successes"] == sum(row["strict_task_feasible"] for row in rows)
             and abs(metrics["mae_ml"] - sum(row["absolute_error_ml"] for row in rows) / len(rows)) < 1e-9
             and abs(metrics["worst_absolute_error_ml"] - max(row["absolute_error_ml"] for row in rows)) < 1e-9,
             "Adaptive task metrics differ from all retained cases")
    for entry in manifest["plots"]:
        raw = inside(project, entry["file"]).read_bytes()
        need(raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == entry["sha256"], "Selected real plot changed")
    need({entry["file"] for entry in manifest["plots"]} == {PLOT_PUBLIC + name for name in PLOT_NAMES},
         "Unexpected selected PNG set")
    private_paths((project / REPORT).read_bytes())
    private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
    return {"sources": 34, "research_models": 1, "collection_cases": 60, "goal_development_cases": 10,
            "terminal_goal_policy_acceptance_proven": False, "default_gui_deployment": False}
