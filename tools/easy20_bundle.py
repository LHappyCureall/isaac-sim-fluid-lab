"""Additive fixed20mL beginner-study publication, separate from strict studies.

All twenty live cases, independent audit, frozen models, selected JSON and one
real figure are required. No SDK, assets, NPZ, raw logs or GUI defaults enter
the explicit versioned source bundle. Existing170 targets remain untouched.
"""

import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


RUN = "droplet_out/pour_policy_easy20/easy20_001"
HERE = "projects/pour_policy_easy20/"
PUBLIC = "research/easy20_20261128/"
REPORT = "results/easy20_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/easy20_20261128/easy20_results.png"
KIND = "easy20_fixed_target_offset_search_v1"
TASK_KIND = "fixed20ml_absolute1ml_24s_onegrain_v1"
COUNTS = {"train": 6, "validation": 4, "test": 10}
SEEDS = {"train": (194101, 194102), "validation": (194201, 194202), "test": tuple(range(194301, 194306))}
NEW_RUNTIME = (HERE + "run_easy_search.py", HERE + "task_score.py")
POSTHOC_NAMES = tuple(HERE + name for name in
    ("select_stage.py", "audit_easy_run.py", "audit_easy_experiment.py", "curate_easy_results.py", "plot_easy_results.py", "test_task_score.py", "test_curate_easy_results.py", "test_audit_easy_run.py"))
# Filled only after each new self-written tool is finalized and reviewed.
REVIEWED_TOOLS = {
    HERE + "select_stage.py": "0941bd58646e7b3d50dbe1ade2c0c328ba2229c73472df6342a5fc86cc6b5ed8",
    HERE + "audit_easy_run.py": "3ba8464416df2d7d02e9b533c11dd0f599dc76da7de5c809db791d3020709855",
    HERE + "audit_easy_experiment.py": "062fecc638d169a1fb0603491ac8c873c593c708e58ebd9ba15a76b0e8d9530e",
    HERE + "curate_easy_results.py": "eabc43d2b9647fd0be44d02f9331324e101ce27a9db4b2c6f2ed14d61a913368",
    HERE + "plot_easy_results.py": "5d1398062892b3853d60c429c74a82267d165ec7649b564754e5a0f7494dfa21",
    HERE + "test_task_score.py": "5b4d3c03bbed75b3037aea700ca58f58261304b5067696c8ffab245a9c1fa4e0",
    HERE + "test_curate_easy_results.py": "93a87d27ab65306a93a136ca689ed028c39d2e6b0080e6c89d2545dc5be7ba23",
    HERE + "test_audit_easy_run.py": "90ad4adcd9e7d77486c26048abe11bac4f6705289b43186c0fb591a7a36cd047",
}
MODELS = ("baseline", "candidate", "recommended")
ATTRIBUTES = ("# Preserve original frozen study bytes, models and provenance.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n"
    "*.py -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "models/*.json -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "experiment_plan.json -text\ntrain_matrix.json -text\n")
README = ("# Fixed20mL beginner research bundle\n\n"
    "This separate task uses20mL +/-1mL,24 raw physical seconds, original-home stability and at most one spilled particle. "
    "The earlier +/-0.1% task and all its failed outcomes remain unchanged. Consult "
    "[the selected actual results](../../results/easy20_20261128.json).\n\n"
    "Six training episodes calibrate one internal coarse stop offset from a declared finite set. Four validation "
    "episodes choose between candidate and baseline, and ten test episodes forming five pairs evaluate the frozen models. "
    "The baseline can already be feasible, can win validation, or can have the same parameter as the candidate. "
    "None of these observations is a new neural-policy improvement or arbitrary-goal demonstration.\n\n"
    "Two new runtime files are included. The other26 runtime dependencies are referenced, without duplicate copies, "
    "through the unchanged V12/V13 manifests. Reconstruct their original relative layout across these archives before "
    "any independent run. Their original/published SHA are listed in source_manifest.json. The existing robot.py "
    "derivative retains its reviewed two resource-path conversions; new source/model bytes are unchanged. "
    "The selector, independent auditor, curator, plotter, CPU test and preregistered plan are separate roles.\n\n"
    "No SDK, third-party robot assets, robot-path or liquid NPZ, full trajectories, video or credentials are included. "
    "Prepare the licensed robot resources and local geometric inputs separately. Robot caches are geometry, not water "
    "replay. Portable paths/rebuilt inputs require a new protocol and engineering validation; do not alter literal guards "
    "or resume the author's private journals. This research bundle is not an out-of-box run or a default GUI upgrade.\n")


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def load_base(project):
    spec = importlib.util.spec_from_file_location("easy20_base_publication_rules", project / "tools/adaptive_response_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_curator(workspace):
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "New beginner publication tools still await final review")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64
             and digest((workspace / name).read_bytes()) == expected, "Reviewed beginner tool changed: " + name)
    spec = importlib.util.spec_from_file_location("easy20_checked_publication_curator", workspace / HERE / "curate_easy_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(workspace, project):
    # Refuse unfinished stages before journals or test data are decoded.
    for stage, count in COUNTS.items():
        summary, job = read(workspace / RUN / stage / "summary.json"), read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and job.get("state") == "complete" and job.get("pid") is None,
             "Beginner publication requires complete6/4/10 live stages")
    base, curator = load_base(project), reviewed_curator(workspace)
    audit_path = workspace / RUN / "overall_audit.json"
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True
         and audit.get("kind") == KIND and audit.get("budgets") == {**COUNTS, "total": 20},
         "Complete independent20-case audit required")
    need(audit.get("auditor_source_file") == HERE + "audit_easy_experiment.py"
         and audit.get("auditor_source_sha256") == REVIEWED_TOOLS[HERE + "audit_easy_experiment.py"],
         "Overall beginner audit is not the reviewed independent implementation")
    expected = curator.curate(workspace / RUN, audit_path, digest(audit_path.read_bytes()))
    selected = read(project / REPORT)
    base.private_paths((project / REPORT).read_bytes())
    ignored = {"record_created_at_utc", "publication", "plots"}
    need({key: value for key, value in selected.items() if key not in ignored}
         == {key: value for key, value in expected.items() if key not in ignored},
         "Selected beginner report differs from all twenty independently audited cases")
    return base, selected, audit


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base, selected, audit = gate(workspace, project)
    prior = dict(prior_prepared)
    parent_public = base.PUBLIC
    original_parent = prior[parent_public + "source_manifest.json"]
    parent = json.loads(original_parent)
    prefix_manifest_bytes = prior[parent_public + "prefix_source_manifest.json"]
    prefix_manifest = json.loads(prefix_manifest_bytes)
    parent_entries = {entry["original_file"]: entry for entry in parent["files"] if entry["role"] == "collection_runtime"}
    parent_entries.update({entry["original_file"]: entry for entry in prefix_manifest["files"] if entry["role"] == "prefix_runtime"})
    old_collector = "projects/pour_policy_adaptive_learning/run_prefix_calibration.py"
    old_collection = "projects/pour_policy_adaptive_learning/collect_cycles.py"
    shared = set(parent_entries) - {old_collector, old_collection}
    train = read(workspace / RUN / "train/protocol.json")
    source_sha = train["source_sha256"]
    need(len(shared) == 26 and set(source_sha) == shared | set(NEW_RUNTIME) and len(source_sha) == 28,
         "Beginner runtime must retain exact26 shared sources plus the new factory and scoring module")
    for stage in COUNTS:
        plan = read(workspace / RUN / stage / "protocol.json")
        need(plan["source_sha256"] == source_sha and plan["input_sha256"] == train["input_sha256"]
             and plan["gui_sha256"] == train["gui_sha256"], "Stage physics/source/input/default profile changed")
    shared_bindings = []
    for name in sorted(shared):
        entry = parent_entries[name]
        need(source_sha[name] == entry["original_sha256"]
             and digest(prior[entry["published_file"]]) == entry["published_sha256"],
             "A shared beginner dependency differs from the historical archive: " + name)
        shared_bindings.append(entry)
    files, mappings = [], []

    def add(original_name, path, role, expected_sha, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected_sha, "Explicit beginner source/model changed: " + original_name)
        published = base.robot_bytes(original_name, raw)
        base.private_paths(published)
        destination = destination or PUBLIC + original_name
        if destination.endswith(".py"):
            ast.parse(published.decode("utf8"), filename=destination)
        files.append((destination, published))
        mappings.append({"role": role, "original_file": original_name, "published_file": destination,
                         "original_sha256": expected_sha, "published_sha256": digest(published),
                         "transform": "two_robot_resource_paths" if original_name == base.ROBOT else "unchanged_raw_bytes"})

    for name in NEW_RUNTIME:
        expected = source_sha[name]
        add(name, workspace / RUN / "train/source_snapshot" / name, "runtime", expected)
    for name, expected in REVIEWED_TOOLS.items():
        add(name, workspace / name, "posthoc_selector_audit_or_test", expected)
    for name, source in (("experiment_plan.json", workspace / RUN / "experiment_plan.json"),
                         ("train_matrix.json", workspace / RUN / "train/matrix.json")):
        add(name, source, "preregistered_plan_or_schedule", digest(source.read_bytes()), PUBLIC + name)
    for role in MODELS:
        evidence = selected["checkpoints"][role]
        path = base.inside(workspace / RUN, evidence["file"])
        add(evidence["file"], path, "frozen_research_" + role, evidence["sha256"], PUBLIC + "models/" + role + ".json")
    plot = read(workspace / PLOTS / "plot_provenance.json")
    need(plot.get("complete") is True and plot.get("synthetic_fixture") is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_easy_results.py"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_easy_results.py"],
         "Final beginner plot provenance differs")
    plot_report = read(workspace / PLOTS / "easy20_results.json")
    need({k: v for k, v in plot_report.items() if k not in ignored_keys()}
         == {k: v for k, v in selected.items() if k not in ignored_keys()}, "Plot input differs from selected results")
    image = (workspace / PLOTS / "easy20_results.png").read_bytes()
    need(image.startswith(b"\x89PNG\r\n\x1a\n")
         and digest(image) == plot["files_sha256"]["easy20_results.png"]
         and digest((workspace / PLOTS / "easy20_results.json").read_bytes()) == plot["files_sha256"]["easy20_results.json"],
         "Final real beginner figure/input bytes changed")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "fixed20_beginner_research_publication_v1",
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "experiment_plan_sha256": selected["experiment_plan_sha256"], "budget": {**COUNTS, "total": 20},
        "source_count": 28, "shared_source_count": 26, "source_sha256": source_sha,
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mappings,
        "shared_source_bindings": shared_bindings,
        "parent_adaptive_manifest_sha256": digest(original_parent),
        "parent_prefix_manifest_sha256": digest(prefix_manifest_bytes),
        "parent_manifests_unchanged": True, "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}],
        "gui_default_sha256_unchanged": train["gui_sha256"],
        "original_strict_standard_unchanged": True, "default_gui_deployment": False,
        "all_continuous_goals_guaranteed": False, "raw_logs_or_assets_distributed": False}
    raw_manifest = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw_manifest)
    files.extend(((PUBLIC + "source_manifest.json", raw_manifest), (PUBLIC + "README.md", README.encode()),
                  (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len({name for name, _ in files}) == len(files), "Duplicate beginner publication destination")
    need(not set(name for name, _ in files) & set(prior), "Beginner publication may not replace any historical target")
    return files


def ignored_keys():
    return {"record_created_at_utc", "publication", "plots"}


def score(rows):
    need(rows, "Empty beginner candidate group")
    if all(row["easy_feasible"] for row in rows):
        return (0, sum(row["physical_action_seconds"] for row in rows) / len(rows))
    return (1, sum(row["absolute_error_ml"] for row in rows) / len(rows),
            sum(row["peak_all_label_spill_particles"] for row in rows))


def metric_check(metric, rows):
    need(rows, "Empty beginner metric group")
    expected = {"cases": len(rows), "joint_passes": sum(row["easy_feasible"] for row in rows),
        "original_strict_joint_passes": sum(row["original_strict_feasible"] for row in rows),
        "mean_absolute_error_ml": sum(row["absolute_error_ml"] for row in rows) / len(rows),
        "maximum_absolute_error_ml": max(row["absolute_error_ml"] for row in rows),
        "mean_signed_error_ml": sum(row["error_ml"] for row in rows) / len(rows),
        "mean_physical_action_seconds": sum(row["physical_action_seconds"] for row in rows) / len(rows),
        "maximum_physical_action_seconds": max(row["physical_action_seconds"] for row in rows),
        "mean_compute_wall_seconds_including_post": sum(row["compute_wall_seconds_including_post"] for row in rows) / len(rows),
        "mean_action_compute_wall_seconds": sum(row["action_compute_wall_seconds"] for row in rows) / len(rows),
        "maximum_peak_all_label_spill_particles": max(row["peak_all_label_spill_particles"] for row in rows),
        "full_home_within_deadline_cases": sum(row["full_home_within_deadline"] for row in rows),
        "no_late_dose_change_cases": sum(row["no_late_dose_change"] for row in rows)}
    need(set(metric) == set(expected), "Beginner metric schema changed")
    for key, value in expected.items():
        need(type(metric[key]) in (int, float) and math.isfinite(metric[key])
             and abs(metric[key] - value) < 1e-8, "Beginner aggregate differs: " + key)


def verify_public(project):
    """Independent clone-safe checks; never needs the private workspace/SDK."""
    project = Path(project).resolve()
    base = load_base(project)
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    need(selected.get("complete") is True and selected.get("kind") == KIND
         and selected.get("budget") == {**COUNTS, "total": 20}, "Expected completed beginner20-case selected result")
    need(manifest.get("schema_version") == 1 and manifest.get("kind") == "fixed20_beginner_research_publication_v1"
         and manifest["report_file"] == REPORT and manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["budget"] == selected["budget"] and manifest["source_count"] == 28
         and manifest["shared_source_count"] == 26, "Beginner report/runtime manifest differs")
    need(manifest["parent_manifests_unchanged"] is True
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes())
         and manifest["parent_prefix_manifest_sha256"] == digest((project / base.PUBLIC / "prefix_source_manifest.json").read_bytes()),
         "Beginner publication changed historical V12/V13 provenance")
    files = manifest["files"]
    runtime = [entry for entry in files if entry["role"] == "runtime"]
    tools = [entry for entry in files if entry["role"] == "posthoc_selector_audit_or_test"]
    plans = [entry for entry in files if entry["role"] == "preregistered_plan_or_schedule"]
    models = [entry for entry in files if entry["role"].startswith("frozen_research_")]
    need(len(runtime) == 2 and {entry["original_file"] for entry in runtime} == set(NEW_RUNTIME)
         and set(manifest["source_sha256"]) == {entry["original_file"] for entry in manifest["shared_source_bindings"]} | set(NEW_RUNTIME)
         and {entry["original_file"] for entry in tools} == set(POSTHOC_NAMES)
         and len(plans) == 2 and {entry["original_file"] for entry in plans} == {"experiment_plan.json", "train_matrix.json"}
         and len(models) == 3, "Explicit beginner file allowlist changed")
    expected_files = {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    for entry in files:
        destination = entry["published_file"]
        need(destination.startswith(PUBLIC) and (project / destination).resolve().is_relative_to(project / PUBLIC),
             "Published beginner destination escaped its versioned root")
        raw = (project / destination).read_bytes()
        need(digest(raw) == entry["published_sha256"], "Published beginner file bytes changed: " + destination)
        if entry["original_file"] == base.ROBOT:
            need(entry["original_sha256"] == base.ROBOT_ORIGINAL and entry["published_sha256"] == base.ROBOT_PUBLIC
                 and entry["transform"] == "two_robot_resource_paths", "Portable robot derivative changed")
        else:
            need(entry["original_sha256"] == entry["published_sha256"]
                 and entry["transform"] == "unchanged_raw_bytes", "Unreviewed beginner source transformation")
        if entry["role"] == "runtime":
            need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime original SHA differs")
        if entry["role"] == "posthoc_selector_audit_or_test":
            need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed beginner tool SHA differs")
        base.private_paths(raw)
        if destination.endswith(".py"):
            compile(raw.decode("utf8"), destination, "exec")
        expected_files.add(destination)
    actual_files = {path.relative_to(project).as_posix() for path in (project / PUBLIC).rglob("*")
                    if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc")}
    need(actual_files == expected_files, "Extra SDK/asset/log/cache or missing file in beginner bundle")
    for entry in manifest["shared_source_bindings"]:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"],
             "Historical shared dependency changed")
    need(len(manifest["shared_source_bindings"]) == 26, "Expected exact26 shared runtime dependencies")
    contract = selected["task_contract"]
    need(contract == {"target_ml": 20., "beginner_absolute_tolerance_ml": 1., "deadline_physical_seconds": 24.,
        "maximum_all_label_peak_spill_particles": 1, "home_quiet_frames": 12,
        "post_observation_physical_seconds": 2., "original_strict_relative_tolerance": .001,
        "original_strict_absolute_tolerance_ml": .02, "original_strict_standard_unchanged": True},
        "Beginner result changed the archived strict task or its own declared criterion")
    rows = selected["cases"]
    need(len(rows) == 20 and len({row["episode_id"] for row in rows}) == 20, "All twenty unique attempts required")
    for stage, count in COUNTS.items():
        group = [row for row in rows if row["stage"] == stage]
        need(len(group) == count and {row["seed"] for row in group} == set(SEEDS[stage]), "Stage budget/seeds differ")
    for row in rows:
        target, amount, elapsed = row["target_ml"], row["received_ml"], row["physical_action_seconds"]
        need(target == 20. and all(type(value) in (int, float) and math.isfinite(value)
             for value in (amount, elapsed, row["error_ml"], row["absolute_error_ml"], row["compute_wall_seconds_including_post"])),
             "Actual fixed-target measurement is not finite")
        need(abs(row["error_ml"] - (amount - target)) < 1e-9
             and abs(row["absolute_error_ml"] - abs(amount - target)) < 1e-9,
             "Actual goal or error was shifted by an internal stop offset")
        safety = bool(row["mass_conserved"] and row["all_states_finite"] and row["full_home_within_deadline"]
            and elapsed <= 24. and row["peak_all_label_spill_particles"] <= 1 and row["no_late_dose_change"])
        need(row["easy_feasible"] is bool(safety and 19. <= amount <= 21.)
             and row["original_strict_dose_passed"] is (19.98 <= amount <= 20.02)
             and row["original_strict_feasible"] is bool(safety and 19.98 <= amount <= 20.02),
             "Primary/secondary strict closed-interval task gates differ")
        need(abs(row["post_physical_seconds"] - 2.) < 1e-8
             and row["compute_wall_seconds_including_post"] > 0, "Post/compute timing differs")
    need(selected["selection"]["validation_only"] is True and selected["selection"]["test_used_for_selection"] is False,
         "Test outcomes were used for model selection")
    for stage in COUNTS:
        roles = {row["role"] for row in rows if row["stage"] == stage}
        need(set(selected["stage_metrics"][stage]) == roles, "Metrics omit a declared stage/role")
        for stage_role in roles:
            metric_check(selected["stage_metrics"][stage][stage_role],
                         [row for row in rows if row["stage"] == stage and row["role"] == stage_role])
    checkpoints = selected["checkpoints"]
    need(set(checkpoints) == set(MODELS), "All three frozen beginner checkpoints required")
    decoded = {}
    for role in MODELS:
        path = project / PUBLIC / "models" / (role + ".json")
        need(digest(path.read_bytes()) == checkpoints[role]["sha256"], "Beginner model raw SHA differs")
        decoded[role] = read(path)
        model = decoded[role]
        need(model["task_kind"] == TASK_KIND and model["policy"].keys() == {"control_target_offset_ml"}
             and type(model["policy"]["control_target_offset_ml"]) in (int, float)
             and model["policy"]["control_target_offset_ml"] in (0., .5, 1.), "Unknown beginner policy kind/parameter")
        if role != "recommended":
            need(model["kind"] == "easy20_internal_stop_offset_v1" and model["role"] == role
                 and model["source_sha256"] == manifest["source_sha256"]
                 and model["is_neural_policy"] is False and model["default_gui_deployment"] is False,
                 "Beginner checkpoint was confused with an old/neural/default policy")
            need(model["training_journal_sha256"] == selected["stage_artifact_sha256"]["train"]["episodes.jsonl"],
                 "Beginner model training journal changed")
        else:
            need(model["kind"] == "easy20_validation_recommendation_v1"
                 and model["test_used_for_selection"] is False
                 and model["validation_journal_sha256"] == selected["stage_artifact_sha256"]["validation"]["episodes.jsonl"],
                 "Wrong recommendation schema/source")
        need(model["selection_tool_sha256"] == REVIEWED_TOOLS[HERE + "select_stage.py"], "Selection implementation differs")
    role = selected["selection"]["recommended_role"]
    need(role in ("baseline", "candidate") and decoded["recommended"]["recommended_role"] == role
         and decoded["recommended"]["checkpoint_sha256"] == checkpoints[role]["sha256"]
         and decoded["recommended"]["policy"] == decoded[role]["policy"], "Validation recommendation differs")
    candidate_offset = decoded["candidate"]["policy"]["control_target_offset_ml"]
    training_groups = {offset: [row for row in rows if row["stage"] == "train" and row["control_target_offset_ml"] == offset]
                       for offset in (0., .5, 1.)}
    need(all(len(group) == 2 and {row["seed"] for row in group} == set(SEEDS["train"])
             for group in training_groups.values()), "All declared training offsets require both seeds")
    expected_offset = min(training_groups, key=lambda offset: (score(training_groups[offset]), (0., .5, 1.).index(offset)))
    need(candidate_offset == expected_offset, "Candidate was not chosen from training alone")
    validation_groups = {name: [row for row in rows if row["stage"] == "validation" and row["role"] == name]
                         for name in ("baseline", "candidate")}
    need(role == min(validation_groups, key=lambda name: (score(validation_groups[name]), ("baseline", "candidate").index(name))),
         "Recommendation differs from validation-only ranking/tie order")
    metric_check(selected["recommended_test_metrics"], [row for row in rows if row["stage"] == "test" and row["role"] == role])
    need(decoded["baseline"]["policy"]["control_target_offset_ml"] == 0.
         and candidate_offset == selected["selection"]["candidate_offset_ml"]
         and selected["candidate_and_baseline_offsets_identical"] is (candidate_offset == 0.), "Candidate/baseline relation differs")
    for row in rows:
        if row["stage"] in ("validation", "test"):
            need(row["control_target_offset_ml"] == (0. if row["role"] == "baseline" else candidate_offset), "Frozen test action changed")
            need(datetime.fromisoformat(decoded[row["role"]]["created_at_utc"]) < datetime.fromisoformat(row["started_at_utc"]),
                 "Role checkpoint was not frozen before evaluation")
        if row["stage"] == "test":
            need(datetime.fromisoformat(decoded["recommended"]["created_at_utc"]) < datetime.fromisoformat(row["started_at_utc"]),
                 "Recommendation was not frozen before test")
    pairs = selected["test_pairs"]
    need(len(pairs) == 5 and {pair["seed"] for pair in pairs} == set(SEEDS["test"]), "Five paired independent conditions required")
    row_lookup = {row["episode_id"]: row for row in rows}
    for pair in pairs:
        for role in ("baseline", "candidate"):
            need(pair[role] == row_lookup[pair[role]["episode_id"]]
                 and pair[role]["stage"] == "test" and pair[role]["role"] == role
                 and pair[role]["seed"] == pair["seed"], "Paired result changed")
    need(selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and manifest["default_gui_deployment"] is False and manifest["original_strict_standard_unchanged"] is True
         and manifest["raw_logs_or_assets_distributed"] is False, "Beginner task was promoted to unsupported acceptance")
    for entry in manifest["plots"]:
        raw = (project / entry["file"]).read_bytes()
        need(entry["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n")
             and digest(raw) == entry["sha256"], "Published real beginner figure changed")
    need(len(manifest["plots"]) == 1, "Exactly one selected beginner figure")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
