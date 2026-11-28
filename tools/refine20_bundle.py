"""Append only a completed independently audited single-correction study.

The existing189 publication targets and their original manifests stay intact.
This bundle references25 frozen V12 sources and adds no SDK, caches or logs.
"""

import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path


RUN = "droplet_out/pour_policy_refine20/refine20_001"
HERE = "projects/pour_policy_refine20/"
PUBLIC = "research/refine20_20261128/"
REPORT = "results/refine20_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/refine20_20261128/refine20_results.png"
KIND = "refine20_settled_single_cycle_angle_v1"
TASK_KIND = "fixed20ml_absolute05ml_24s_onegrain_v1"
COUNTS = {"engineering": 3, "train": 6, "validation": 4, "test": 10}
SEEDS = {"engineering": (195001,), "train": (195101, 195102),
         "validation": (195201, 195202), "test": tuple(range(195301, 195306))}
ANGLES = (84., 84.1, 84.2)
NEW_RUNTIME = tuple(HERE + name for name in
                    ("refine_control.py", "run_refine_search.py", "task_score.py"))
POSTHOC_NAMES = tuple(HERE + name for name in
    ("select_stage.py", "audit_refine_run.py", "audit_refine_experiment.py",
     "curate_refine_results.py", "plot_refine_results.py", "test_task_score.py",
     "test_refine_control.py", "test_audit_refine_run.py", "test_curate_refine_results.py"))
# Each separate posthoc implementation is reviewed and frozen explicitly.
REVIEWED_TOOLS = {
    HERE + "select_stage.py": "1b2cb90009e97289003f9983bfb1e4e67153ac2b6d1bbea6bbe5674d7bdf56fe",
    HERE + "audit_refine_run.py": "ebd78213bb847b0db80e2d9e18e17f35320c08512313e5bb25208b6163172e9a",
    HERE + "audit_refine_experiment.py": "905bc8b09b2a193c94d6fae3478baefd8acfd664af55f3b247f401890a1687d1",
    HERE + "curate_refine_results.py": "9a2a105d0729067c99c65cdd40d9cea213ef775b1763a9e39aa88fc0df1db80f",
    HERE + "plot_refine_results.py": "5f12ee91d9270a8c2d8ee42853107e84dcbd6610e9825d3843a58b68fc38dfe3",
    HERE + "test_task_score.py": "137580213ea217c2a2841177b75a3655896b79e642e47c1cfd291922a9ab73ae",
    HERE + "test_refine_control.py": "5917c1d2bd26f53912e2db762a987410f9c77c364d9a20f71bc8140df0f6e2a2",
    HERE + "test_audit_refine_run.py": "b517a031792b8bc9de5a6f986043c341b8f0ba276f1982cff9c761998d9b3c6e",
    HERE + "test_curate_refine_results.py": "75dc61114439169dbcc421287e16f1343b89eb93e1b04613a215864f12173ea0",
}
MODELS = ("baseline", "candidate", "recommended")
ATTRIBUTES = ("# Preserve the frozen study source/model bytes and their SHA.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n"
    "*.py -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "models/*.json -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "experiment_plan.json -text\nengineering_matrix.json -text\ntrain_matrix.json -text\n")
README = ("# Fixed20mL measured-correction research bundle\n\n"
    "The primary curriculum is20mL +/-0.5mL,24 raw physical seconds through original-home stability, "
    "and at most one all-label peak spilled particle. The earlier +/-1mL and original +/-0.020mL "
    "criteria are separate secondary results. See [the actual selected results](../../results/refine20_20261128.json).\n\n"
    "Three engineering episodes gate safety only. Six training episodes calibrate one absolute approach angle, "
    "four validation episodes choose candidate or the same-path coarse-only reference, and ten test episodes "
    "form five matched initial-state pairs. All motion speeds, coarse parameters and micro-action settings "
    "are fixed. At most one complete correction cycle is measured after actual quiet; a model prediction "
    "or video playback never supplies the final volume. Test outcomes cannot choose or change the model.\n\n"
    "This applies the settled-feedback principle of precision V2 on a different fixed-mouth trajectory. "
    "It is neither the original V2 controller nor a paired comparison with historical V2 results. "
    "Finite angle calibration is not PPO or a neural policy, and fixed20mL results do not establish arbitrary "
    "20--40mL or hardware accuracy. Kinematic pregrasp and particle-mass truth are not contact grasp or a real scale.\n\n"
    "Three new runtime sources are archived unchanged. The other25 runtime sources are referenced through "
    "the original V12 manifest without duplication; prefix_control.py is not a runtime dependency of this study. "
    "Reconstruct the original module layout across these archives before an independent run. The existing "
    "robot.py derivative retains its reviewed two resource-path conversions; both SHA are listed explicitly. "
    "Preparation of licensed robot resources and geometric caches requires a new protocol and engineering "
    "validation. Do not rebind literal guards or resume the author's private journals.\n\n"
    "Posthoc audits also require the prior V12/V13 archive references "
    "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py, audit_response_collection.py "
    "and cycle_samples.py, plus the earlier V9 reference projects/pour_policy_training_v9/audit_fine_run.py. "
    "Restore their original relative paths and preserve their literal SHA guards; these files are not copied again here.\n\n"
    "The selector, independent auditors, curator, plotter, CPU tests, plans and three frozen research JSON "
    "checkpoints have separate roles. No SDK, third-party assets, NPZ, full trajectories, videos or credentials "
    "are included. This archive does not replace any GUI default or provide an out-of-box GUI launch.\n")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def load(project, filename, name):
    spec = importlib.util.spec_from_file_location(name, project / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_curator(workspace):
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "Refine20 posthoc allowlist differs")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64
             and digest((workspace / name).read_bytes()) == expected,
             "Final reviewed refine20 tool required: " + name)
    return load(workspace, HERE + "curate_refine_results.py", "publication_refine20_curator")


def ignored_keys():
    return {"record_created_at_utc", "publication", "plots"}


def same_summary(left, right):
    return {k: v for k, v in left.items() if k not in ignored_keys()} == {
        k: v for k, v in right.items() if k not in ignored_keys()}


def gate(workspace, project):
    # Reject partial experiments before reading journals or held-out cases.
    for stage, count in COUNTS.items():
        summary = read(workspace / RUN / stage / "summary.json")
        job = read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and job.get("state") == "complete" and job.get("pid") is None,
             "Refine20 publication requires complete3/6/4/10 live stages")
    base = load(project, "tools/adaptive_response_bundle.py", "refine20_parent_publication")
    curator = reviewed_curator(workspace)
    audit_path = workspace / RUN / "overall_audit.json"
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True
         and audit.get("kind") == KIND and audit.get("budgets") == {**COUNTS, "total": 23},
         "Complete independent23-case audit required")
    need(audit.get("auditor_source_file") == HERE + "audit_refine_experiment.py"
         and audit.get("auditor_source_sha256") == REVIEWED_TOOLS[HERE + "audit_refine_experiment.py"],
         "Overall audit must use the reviewed independent implementation")
    selected = read(project / REPORT)
    base.private_paths((project / REPORT).read_bytes())
    expected = curator.build_summary(workspace / RUN, audit_path, digest(audit_path.read_bytes()))
    need(same_summary(selected, expected), "Selected refine20 report differs from audited23 actual cases")
    return base, selected, audit


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base, selected, audit = gate(workspace, project)
    prior = dict(prior_prepared)
    parent_bytes = prior[base.PUBLIC + "source_manifest.json"]
    parent = json.loads(parent_bytes)
    entries = {entry["original_file"]: entry for entry in parent["files"]
               if entry["role"] == "collection_runtime"}
    shared = set(entries) - {"projects/pour_policy_adaptive_learning/collect_cycles.py"}
    train = read(workspace / RUN / "train/protocol.json")
    source_sha = train["source_sha256"]
    need(len(shared) == 25 and len(source_sha) == 28
         and set(source_sha) == shared | set(NEW_RUNTIME),
         "Expected exact25 shared V12 sources and three new runtime files")
    for stage in COUNTS:
        plan = read(workspace / RUN / stage / "protocol.json")
        need(plan["source_sha256"] == source_sha and plan["input_sha256"] == train["input_sha256"]
             and plan["gui_sha256"] == train["gui_sha256"], "Refine20 stages changed sources/physics/defaults")
    shared_bindings = []
    for name in sorted(shared):
        entry = entries[name]
        need(source_sha[name] == entry["original_sha256"]
             and digest(prior[entry["published_file"]]) == entry["published_sha256"],
             "Shared historical dependency changed: " + name)
        shared_bindings.append(entry)
    files, mappings = [], []

    def add(original, path, role, expected, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected, "Explicit refine20 file changed: " + original)
        base.private_paths(raw)
        destination = destination or PUBLIC + original
        if destination.endswith(".py"):
            ast.parse(raw.decode("utf8"), filename=destination)
        files.append((destination, raw))
        mappings.append({"role": role, "original_file": original, "published_file": destination,
            "original_sha256": expected, "published_sha256": digest(raw), "transform": "unchanged_raw_bytes"})

    for name in NEW_RUNTIME:
        add(name, workspace / RUN / "train/source_snapshot" / name, "runtime", source_sha[name])
    for name, expected in REVIEWED_TOOLS.items():
        add(name, workspace / name, "posthoc_selector_audit_or_test", expected)
    for name, source in (("experiment_plan.json", workspace / RUN / "experiment_plan.json"),
                         ("engineering_matrix.json", workspace / RUN / "engineering/matrix.json"),
                         ("train_matrix.json", workspace / RUN / "train/matrix.json")):
        add(name, source, "preregistered_plan_or_schedule", digest(source.read_bytes()), PUBLIC + name)
    for role in MODELS:
        evidence = selected["checkpoints"][role]
        add(evidence["file"], base.inside(workspace / RUN, evidence["file"]),
            "frozen_research_" + role, evidence["sha256"], PUBLIC + "models/" + role + ".json")
    plot = read(workspace / PLOTS / "plot_provenance.json")
    need(plot.get("complete") is True and plot.get("synthetic_fixture") is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_refine_results.py"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_refine_results.py"],
         "Final refine20 plot provenance differs")
    plot_report_path = workspace / PLOTS / "refine20_results.json"
    need(same_summary(read(plot_report_path), selected), "Final figure did not use the selected actual results")
    image = (workspace / PLOTS / "refine20_results.png").read_bytes()
    need(image.startswith(b"\x89PNG\r\n\x1a\n")
         and digest(image) == plot["files_sha256"]["refine20_results.png"]
         and digest(plot_report_path.read_bytes()) == plot["files_sha256"]["refine20_results.json"],
         "Final real figure or its bound result bytes changed")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "refine20_single_cycle_research_publication_v1",
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "experiment_plan_sha256": selected["experiment_plan_sha256"], "budget": {**COUNTS, "total": 23},
        "source_count": 28, "shared_source_count": 25, "source_sha256": source_sha,
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mappings,
        "shared_source_bindings": shared_bindings, "parent_adaptive_manifest_sha256": digest(parent_bytes),
        "parent_manifests_unchanged": True, "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}],
        "gui_default_sha256_unchanged": train["gui_sha256"], "original_strict_standard_unchanged": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "historical_v2_is_not_paired_baseline": True, "raw_logs_or_assets_distributed": False}
    raw_manifest = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw_manifest)
    files.extend(((PUBLIC + "source_manifest.json", raw_manifest), (PUBLIC + "README.md", README.encode()),
                  (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len({name for name, _ in files}) == len(files) == 22, "Refine20 explicit22-file allowlist differs")
    need(not set(name for name, _ in files) & set(prior), "Refine20 may not replace a historical publication target")
    return files


def score(rows):
    need(rows, "Empty refine20 selection group")
    if any(row["valid_evidence"] is not True for row in rows):
        return (2,)
    if all(row["refine_feasible"] for row in rows):
        return (0, sum(row["physical_action_seconds"] for row in rows) / len(rows))
    return (1, sum(row["absolute_error_ml"] for row in rows) / len(rows),
            sum(row["peak_all_label_spill_particles"] for row in rows))


def metric_check(easy, metric, rows):
    need(metric["easy_joint_passes"] == sum(row["easy_feasible"] for row in rows),
         "Secondary +/-1mL count differs")
    need(metric["valid_evidence_cases"] == sum(row["valid_evidence"] for row in rows),
         "Valid physical-evidence count differs")
    easy.metric_check({k: v for k, v in metric.items() if k not in ("easy_joint_passes", "valid_evidence_cases")},
                      [{**row, "easy_feasible": row["refine_feasible"]} for row in rows])


def timestamp(value):
    result = datetime.fromisoformat(value)
    need(result.tzinfo is not None and result.utcoffset() is not None, "Explicit UTC-aware freeze/reset time required")
    return result


def verify_public(project):
    """Clone-safe independent schema/ranking/byte checks; no SDK or private data."""
    project = Path(project).resolve()
    base = load(project, "tools/adaptive_response_bundle.py", "refine20_clone_parent")
    easy = load(project, "tools/easy20_bundle.py", "refine20_clone_metric_reference")
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    need(selected.get("complete") is True and selected.get("kind") == KIND
         and selected.get("budget") == {**COUNTS, "total": 23}, "Completed23-case refine20 report required")
    need(manifest.get("schema_version") == 1 and manifest.get("kind") == "refine20_single_cycle_research_publication_v1"
         and manifest["report_file"] == REPORT and manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["budget"] == selected["budget"] and manifest["source_count"] == 28
         and manifest["shared_source_count"] == 25, "Refine20 manifest/report differs")
    need(manifest["parent_manifests_unchanged"] is True
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes()),
         "Refine20 changed V12 provenance")
    files = manifest["files"]
    expected = {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    role_files = lambda role: {entry["original_file"] for entry in files if entry["role"] == role}
    need(role_files("runtime") == set(NEW_RUNTIME)
         and role_files("posthoc_selector_audit_or_test") == set(POSTHOC_NAMES)
         and role_files("preregistered_plan_or_schedule") == {"experiment_plan.json", "engineering_matrix.json", "train_matrix.json"}
         and len([entry for entry in files if entry["role"].startswith("frozen_research_")]) == 3
         and len(files) == 18, "Refine20 file allowlist changed")
    for entry in files:
        destination = entry["published_file"]
        need(destination.startswith(PUBLIC) and (project / destination).resolve().is_relative_to(project / PUBLIC),
             "Refine20 archive path escaped its versioned root")
        raw = (project / destination).read_bytes()
        need(digest(raw) == entry["published_sha256"] == entry["original_sha256"]
             and entry["transform"] == "unchanged_raw_bytes", "Refine20 frozen source/model bytes changed")
        if entry["role"] == "runtime":
            need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime SHA differs")
        if entry["role"] == "posthoc_selector_audit_or_test":
            need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed posthoc SHA differs")
        base.private_paths(raw)
        if destination.endswith(".py"):
            compile(raw.decode("utf8"), destination, "exec")
        expected.add(destination)
    actual = {path.relative_to(project).as_posix() for path in (project / PUBLIC).rglob("*")
              if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc")}
    need(actual == expected, "Extra SDK/log/asset/cache or missing file in refine20 archive")
    bindings = manifest["shared_source_bindings"]
    need(len(bindings) == 25 and set(manifest["source_sha256"]) == set(NEW_RUNTIME) | {e["original_file"] for e in bindings}
         and not any(entry["original_file"].endswith("prefix_control.py") for entry in bindings),
         "Shared25 source bindings changed")
    for entry in bindings:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Shared V12 dependency changed")
    need(selected["task_contract"] == {"target_ml": 20., "primary_absolute_tolerance_ml": .5,
        "secondary_easy_absolute_tolerance_ml": 1., "deadline_physical_seconds": 24.,
        "maximum_all_label_peak_spill_particles": 1, "home_quiet_frames": 12,
        "post_observation_physical_seconds": 2., "original_strict_relative_tolerance": .001,
        "original_strict_absolute_tolerance_ml": .02, "original_strict_standard_unchanged": True},
        "Primary/secondary task standards changed")
    need(selected["physics"] == {"solver": "live_PBF_action_conditioned", "physics_hz": 60,
        "pressure_iterations": 6, "minimum_substeps": 4, "cuda_graph": False,
        "spacing_m": .0027, "particle_volume_ml": .019683, "total_particles": 11625,
        "source_particles": 2880, "volume_measurement": "fixed-identity particle-mass simulation truth",
        "contact_force_grasp_simulated": False}, "Physics/particle/measurement declaration changed")
    rows = selected["cases"]
    need(len(rows) == 23 and len({row["episode_id"] for row in rows}) == 23, "All23 unique attempts required")
    for stage, count in COUNTS.items():
        group = [row for row in rows if row["stage"] == stage]
        need(len(group) == count and {row["seed"] for row in group} == set(SEEDS[stage]), "Stage budget/seeds differ")
        for role in {row["role"] for row in group}:
            metric_check(easy, selected["stage_metrics"][stage][role], [row for row in group if row["role"] == role])
    for row in rows:
        amount, elapsed = row["received_ml"], row["physical_action_seconds"]
        need(row["split"] == ("development" if row["stage"] == "engineering" else row["stage"]),
             "Engineering/development and train/validation/test split changed")
        if row["stage"] in ("engineering", "train"):
            need(row["correction_enabled"] is True, "Engineering/training must use declared correction candidates")
        need(row["target_ml"] == 20. and all(type(value) in (int, float) and math.isfinite(value)
             for value in (amount, elapsed, row["error_ml"], row["absolute_error_ml"], row["compute_wall_seconds_including_post"]))
             and abs(row["error_ml"] - (amount - 20.)) < 1e-9
             and abs(row["absolute_error_ml"] - abs(amount - 20.)) < 1e-9, "Actual goal/dose/error changed")
        safety = bool(row["valid_evidence"] and row["mass_conserved"] and row["all_states_finite"] and row["full_home_within_deadline"]
            and elapsed <= 24. and row["peak_all_label_spill_particles"] <= 1 and row["no_late_dose_change"])
        need(row["refine_feasible"] is bool(safety and 19.5 <= amount <= 20.5)
             and row["easy_feasible"] is bool(safety and 19. <= amount <= 21.)
             and row["original_strict_dose_passed"] is (19.98 <= amount <= 20.02)
             and row["original_strict_feasible"] is bool(safety and 19.98 <= amount <= 20.02),
             "Three distinct closed-interval task gates differ")
        need(type(row["correction_enabled"]) is bool and row["approach_tilt_deg"] in ANGLES
             and row["actual_correction_cycles_started"] in (0, 1)
             and row["actual_correction_cycles_ended"] == row["actual_correction_cycles_started"]
             and (row["correction_enabled"] or row["actual_correction_cycles_started"] == 0), "One-cycle/baseline mode differs")
        need(len(row["actual_cycles"]) == row["actual_correction_cycles_started"], "Actual cycle detail count differs")
        for cycle in row["actual_cycles"]:
            need(cycle["has_recorded_endpoint"] is True
                 and 0. <= cycle["start_physical_s"] < cycle["end_physical_s"] <= elapsed
                 and abs(cycle["outcome"]["settled_delta_ml"]
                         - (cycle["actual_end_received_ml"] - cycle["actual_start_received_ml"])) < 1e-8,
                 "Actual full-cycle measurement/timing differs")
        need(abs(row["post_physical_seconds"] - 2.) < 1e-8 and row["compute_wall_seconds_including_post"] > 0,
             "Post/compute timing differs")
    need(selected["selection"]["validation_only"] is True and selected["selection"]["test_used_for_selection"] is False
         and selected["engineering_used_for_model_selection"] is False, "Engineering/test entered model selection")
    decoded = {role: read(project / PUBLIC / "models" / (role + ".json")) for role in MODELS}
    for role, model in decoded.items():
        need(digest((project / PUBLIC / "models" / (role + ".json")).read_bytes()) == selected["checkpoints"][role]["sha256"],
             "Frozen model raw SHA differs")
        policy = model["policy"]
        need(model["task_kind"] == TASK_KIND and set(policy) == {"enable_correction", "approach_tilt_deg"}
             and type(policy["enable_correction"]) is bool and policy["approach_tilt_deg"] in ANGLES,
             "Wrong refine20 policy/schema")
        need(model["selection_tool_sha256"] == REVIEWED_TOOLS[HERE + "select_stage.py"], "Selection tool differs")
        if role != "recommended":
            need(model["kind"] == "refine20_settled_angle_v1" and model["role"] == role
                 and model["source_sha256"] == manifest["source_sha256"] and model["is_neural_policy"] is False
                 and model["default_gui_deployment"] is False
                 and model["training_journal_sha256"] == selected["stage_artifact_sha256"]["train"]["episodes.jsonl"]
                 and model["engineering_audit_sha256"] == selected["stage_artifact_sha256"]["engineering"]["audit.json"],
                 "Refine20 model provenance/default claim differs")
        else:
            need(model["kind"] == "refine20_validation_recommendation_v1" and model["test_used_for_selection"] is False
                 and model["validation_journal_sha256"] == selected["stage_artifact_sha256"]["validation"]["episodes.jsonl"],
                 "Wrong validation-only recommendation schema")
    need(decoded["baseline"]["policy"] == {"enable_correction": False, "approach_tilt_deg": 84.}
         and decoded["candidate"]["policy"]["enable_correction"] is True, "Coarse/correction modes changed")
    angle = decoded["candidate"]["policy"]["approach_tilt_deg"]
    groups = {a: [row for row in rows if row["stage"] == "train" and row["approach_tilt_deg"] == a] for a in ANGLES}
    need(all(len(group) == 2 and {row["seed"] for row in group} == set(SEEDS["train"]) for group in groups.values()),
         "All three angles require both training conditions")
    need(angle == min(groups, key=lambda value: (score(groups[value]), ANGLES.index(value))), "Candidate is not train-only selection")
    validation = {role: [row for row in rows if row["stage"] == "validation" and row["role"] == role]
                  for role in ("baseline", "candidate")}
    recommended_role = selected["selection"]["recommended_role"]
    need(recommended_role == min(validation, key=lambda role: (score(validation[role]), ("baseline", "candidate").index(role)))
         and decoded["recommended"]["recommended_role"] == recommended_role
         and decoded["recommended"]["checkpoint_sha256"] == selected["checkpoints"][recommended_role]["sha256"]
         and decoded["recommended"]["policy"] == decoded[recommended_role]["policy"]
         and angle == selected["selection"]["candidate_approach_tilt_deg"], "Validation recommendation differs")
    for row in rows:
        if row["stage"] in ("validation", "test"):
            need(row["correction_enabled"] is (row["role"] == "candidate")
                 and row["approach_tilt_deg"] == decoded[row["role"]]["policy"]["approach_tilt_deg"]
                 and timestamp(decoded[row["role"]]["created_at_utc"]) < timestamp(row["started_at_utc"]),
                 "Role profile was changed or not frozen before evaluation")
        if row["stage"] == "test":
            need(timestamp(decoded["recommended"]["created_at_utc"]) < timestamp(row["started_at_utc"]),
                 "Recommendation did not precede independent tests")
    metric_check(easy, selected["recommended_test_metrics"], [row for row in rows if row["stage"] == "test" and row["role"] == recommended_role])
    pairs, lookup = selected["test_pairs"], {row["episode_id"]: row for row in rows}
    need(len(pairs) == 5 and {pair["seed"] for pair in pairs} == set(SEEDS["test"]), "Five test pairs required")
    for pair in pairs:
        for role in ("baseline", "candidate"):
            need(pair[role] == lookup[pair[role]["episode_id"]] and pair[role]["role"] == role
                 and pair[role]["stage"] == "test" and pair[role]["seed"] == pair["seed"], "Paired test data changed")
    need(selected["all_motion_speeds_fixed"] is True and selected["baseline_is_coarse_only_zero_offset"] is True
         and selected["candidate_is_single_settled_correction"] is True
         and selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and manifest["default_gui_deployment"] is False and manifest["historical_v2_is_not_paired_baseline"] is True
         and manifest["original_strict_standard_unchanged"] is True and manifest["raw_logs_or_assets_distributed"] is False,
         "Unsupported deployment, range, historical V2 or implementation claim")
    need(len(manifest["plots"]) == 1, "Exactly one selected real figure required")
    for image in manifest["plots"]:
        raw = (project / image["file"]).read_bytes()
        need(image["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == image["sha256"],
             "Published real refine20 figure changed")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
