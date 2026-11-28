"""Add the completed fixed conditional-retry experiment without old edits.

One seen development revisit precedes twenty fresh paired tests. No training,
validation, parameter search or test-based strategy selection is performed.
"""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path


RUN = "droplet_out/pour_policy_retry20/retry20_001"
HERE = "projects/pour_policy_retry20/"
PUBLIC = "research/retry20_20261128/"
REPORT = "results/retry20_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/retry20_20261128/retry20_results.png"
OLD_REPORT = "results/refine20_20261128.json"
KIND = "retry20_conditional_second_cycle_v1"
TASK_KIND = "fixed20ml_absolute05ml_24s_onegrain_v1"
COUNTS = {"pilot": 1, "test": 20}
SEEDS = {"pilot": {195301}, "test": set(range(196301, 196311))}
PROFILES = {"baseline": {"enable_retry": False}, "candidate": {"enable_retry": True}}
NEW_RUNTIME = tuple(HERE + name for name in ("retry_control.py", "run_retry_test.py", "task_score.py"))
POSTHOC_NAMES = tuple(HERE + name for name in ("freeze_strategy.py", "audit_retry_run.py",
    "audit_retry_experiment.py", "curate_retry_results.py", "plot_retry_results.py",
    "test_retry_control.py", "test_audit_retry_run.py", "test_curate_retry_results.py"))
# Each posthoc implementation is reviewed and frozen separately from runtime.
REVIEWED_TOOLS = {
    HERE + "freeze_strategy.py": "eddb39c0b8926608eafaf3e101de044842835880202606fc942e094a91310029",
    HERE + "audit_retry_run.py": "f10e0d87cb2e56d18af7c727a4967e25090d8e4ef3bb06fcb966c2ddf8a916fc",
    HERE + "audit_retry_experiment.py": "8ec4c23d77bfceb4bf5f3f2082a8f2572991af96cc9d8c71737be1296fe21084",
    HERE + "curate_retry_results.py": "0b0ec8e9aebf6cc70acb763d0ca03c8e06e44eb838258dc508850dc3ea1dd811",
    HERE + "plot_retry_results.py": "ff8c5aeb803c8f578f482e203d53c1780e261c23edd8033e8e662dc317a839ec",
    HERE + "test_retry_control.py": "637dc4b9b8e246ad4a9394c37da381ae6f68d074477b8018c9e6c99b55eadac2",
    HERE + "test_audit_retry_run.py": "a5259d8b20dc2f0e43a4f1042d1557cdefa727735102f240bf35d00be8652ac4",
    HERE + "test_curate_retry_results.py": "fe3b39b05e60d16db5a3c2640a8195d081ef43b5e18fff10fbfbcbd4182aeb5a",
}
FIXED_PARAMETERS = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
    "return_fraction": 1., "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.,
    "approach_tilt_deg_s": 6., "action_delta_deg": .03, "action_tilt_deg_s": .6,
    "hold_seconds": .1, "backoff_delta_deg": .5, "backoff_tilt_deg_s": 4.,
    "anchor_return_tilt_deg_s": 6., "quiet_seconds": .25, "maximum_measurement_hold_s": 1.,
    "first_approach_tilt_deg": 84., "retry_approach_tilt_deg": 84.1,
    "maximum_cycles_baseline": 1, "maximum_cycles_candidate": 2,
    "target_ml": 20., "absolute_tolerance_ml": .5, "deadline_s": 24., "peak_all_label_spill_limit": 1}
ATTRIBUTES = ("# Preserve frozen strategy/source bytes and their SHA.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n"
    "*.py -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    "models/*.json -text\nexperiment_plan.json -text\npilot_matrix.json -text\n")
README = ("# Conditional second pouring cycle: research bundle\n\n"
    "This experiment targets 20 mL within +/-0.5 mL and 24 raw physical seconds, including return to "
    "the original pregrasp position and 12 stable frames. Peak spill across all particle labels must "
    "not exceed 1 particle. A further 2 seconds of observation checks for late water. The earlier +/-1 mL "
    "and original +/-0.020 mL criteria remain separate secondary results. See "
    "[the actual selected report](../../results/retry20_20261128.json).\n\n"
    "The first approach is fixed at 84.0 degrees. A second approach at 84.1 degrees is permitted only "
    "when the actual settled volume is below 19.5 mL, current peak spill is at most 1 particle, and a "
    "conservative motion budget leaves time to return and stabilize at home. The decision event records "
    "intent; the second `response_cycle_start` and measured endpoint record "
    "actual execution. A complete cycle includes approach drainage, cutoff and quiet measurement: "
    "the second micro action can be skipped while retaining the water already released during approach.\n\n"
    "The known failure at seed 195301 is a development pilot, never an independent test. Its first "
    "cycle is compared with the preserved old trace. The audit checks actual water, motion, partitions "
    "and solver substeps for every completed frame, and retains the documented inherited diagnostic "
    "counter exception at time zero. After the pilot passes, one fixed strategy JSON is frozen before "
    "ten new paired conditions. Baseline means one cycle; candidate means the same first cycle plus "
    "conditional retry. No model is trained, validated, selected or updated from these tests. Ten pairs "
    "with narrow initial-state randomization do not establish arbitrary 20--40 mL or hardware precision.\n\n"
    "Three new runtime sources reference 25 unchanged V12 sources rather than duplicate them. The old "
    "refine20 study supplies only the known failure reference and is not a runtime ancestor. The audit tools "
    "also require the V12/V13 references projects/pour_policy_adaptive_learning/audit_prefix_calibration.py, "
    "audit_response_collection.py and cycle_samples.py, plus the prior V9 reference "
    "projects/pour_policy_training_v9/audit_fine_run.py. Restore original relative paths and preserve "
    "literal SHA guards; do not rewrite them to resume private journals.\n\n"
    "The existing robot.py derivative keeps its two reviewed resource-path conversions. Licensed robot "
    "resources and geometric caches are prepared separately under a new protocol; caches are not liquid "
    "replay. No SDK, third-party assets, NPZ, full logs, videos or credentials are included. Kinematic "
    "pregrasp and particle mass truth are not a contact grasp, wrist scale, or hardware test. This archive "
    "does not replace a GUI default or supply an out-of-box GUI entry.\n")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def load(project, path, name):
    spec = importlib.util.spec_from_file_location(name, project / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_summary(left, right):
    ignored = {"record_created_at_utc", "publication", "plots"}
    return {k: v for k, v in left.items() if k not in ignored} == {k: v for k, v in right.items() if k not in ignored}


def gate(workspace, project):
    for stage, count in COUNTS.items():
        summary, job = read(workspace / RUN / stage / "summary.json"), read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and job.get("state") == "complete" and job.get("pid") is None,
             "Retry20 requires complete1/20 live stages before publication")
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "Retry20 posthoc allowlist differs")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64
             and digest((workspace / name).read_bytes()) == expected, "Final reviewed retry tool required: " + name)
    base = load(project, "tools/adaptive_response_bundle.py", "retry20_parent_publication")
    curator = load(workspace, HERE + "curate_retry_results.py", "retry20_reviewed_curator")
    audit_path = workspace / RUN / "overall_audit.json"
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True
         and audit.get("kind") == KIND and audit.get("budgets") == {**COUNTS, "total": 21}
         and audit["auditor_source_file"] == HERE + "audit_retry_experiment.py"
         and audit["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_retry_experiment.py"],
         "Complete independent reviewed21-case audit required")
    selected = read(project / REPORT)
    expected = curator.build_summary(workspace / RUN, audit_path, digest(audit_path.read_bytes()))
    need(same_summary(selected, expected), "Selected retry report differs from all21 independently audited cases")
    base.private_paths((project / REPORT).read_bytes())
    return base, selected, audit


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base, selected, audit = gate(workspace, project)
    prior = dict(prior_prepared)
    parent_bytes = prior[base.PUBLIC + "source_manifest.json"]
    entries = {entry["original_file"]: entry for entry in json.loads(parent_bytes)["files"]
               if entry["role"] == "collection_runtime"}
    shared = set(entries) - {"projects/pour_policy_adaptive_learning/collect_cycles.py"}
    pilot = read(workspace / RUN / "pilot/protocol.json")
    sources = pilot["source_sha256"]
    need(len(shared) == 25 and len(sources) == 28 and set(sources) == shared | set(NEW_RUNTIME),
         "Retry20 requires exact25 shared sources and three new runtime files")
    test = read(workspace / RUN / "test/protocol.json")
    for key in ("source_sha256", "input_sha256", "gui_sha256"):
        need(pilot[key] == test[key], "Known pilot and fresh tests changed frozen inputs")
    bindings = []
    for name in sorted(shared):
        entry = entries[name]
        need(sources[name] == entry["original_sha256"]
             and digest(prior[entry["published_file"]]) == entry["published_sha256"], "Shared V12 dependency changed")
        bindings.append(entry)
    files, mappings = [], []

    def add(original, path, role, expected, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected, "Explicit retry20 source/strategy changed: " + original)
        base.private_paths(raw)
        destination = destination or PUBLIC + original
        if destination.endswith(".py"):
            ast.parse(raw.decode("utf8"), filename=destination)
        files.append((destination, raw))
        mappings.append({"role": role, "original_file": original, "published_file": destination,
            "original_sha256": expected, "published_sha256": digest(raw), "transform": "unchanged_raw_bytes"})

    for name in NEW_RUNTIME:
        add(name, workspace / RUN / "pilot/source_snapshot" / name, "runtime", sources[name])
    for name, expected in REVIEWED_TOOLS.items():
        add(name, workspace / name, "posthoc_freeze_audit_or_test", expected)
    for name, source in (("experiment_plan.json", workspace / RUN / "experiment_plan.json"),
                         ("pilot_matrix.json", workspace / RUN / "pilot/matrix.json")):
        add(name, source, "preregistered_plan_or_schedule", digest(source.read_bytes()), PUBLIC + name)
    strategy = selected["fixed_strategy"]
    add(strategy["file"], base.inside(workspace / RUN, strategy["file"]), "fixed_research_strategy",
        strategy["sha256"], PUBLIC + "models/fixed_strategy.json")
    plot = read(workspace / PLOTS / "plot_provenance.json")
    report_path = workspace / PLOTS / "retry20_results.json"
    image = (workspace / PLOTS / "retry20_results.png").read_bytes()
    need(plot.get("complete") is True and plot.get("synthetic_fixture") is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_retry_results.py"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_retry_results.py"]
         and same_summary(read(report_path), selected), "Real retry figure provenance differs")
    need(image.startswith(b"\x89PNG\r\n\x1a\n") and digest(image) == plot["files_sha256"]["retry20_results.png"]
         and digest(report_path.read_bytes()) == plot["files_sha256"]["retry20_results.json"], "Real retry plot/input bytes changed")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "retry20_fixed_strategy_research_publication_v1",
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "experiment_plan_sha256": selected["experiment_plan_sha256"], "budget": {**COUNTS, "total": 21},
        "source_count": 28, "shared_source_count": 25, "source_sha256": sources,
        "input_sha256": pilot["input_sha256"],
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mappings,
        "shared_source_bindings": bindings, "parent_adaptive_manifest_sha256": digest(parent_bytes),
        "parent_manifests_unchanged": True, "known_case_reference_report": OLD_REPORT,
        "known_case_reference_report_sha256": digest((project / OLD_REPORT).read_bytes()),
        "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}],
        "gui_default_sha256_unchanged": pilot["gui_sha256"], "original_strict_standard_unchanged": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "no_training_validation_or_selection": True, "pilot_is_seen_development_not_independent": True,
        "raw_logs_or_assets_distributed": False}
    raw_manifest = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw_manifest)
    files.extend(((PUBLIC + "source_manifest.json", raw_manifest), (PUBLIC + "README.md", README.encode()),
                  (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len({name for name, _ in files}) == len(files) == 18, "Retry20 explicit18-file allowlist changed")
    need(not set(name for name, _ in files) & set(prior), "Retry20 may not replace any of211 earlier targets")
    return files


def metric_check(reference, easy, metric, rows):
    extra = {"actual_second_cycle_cases": sum(row["actual_second_cycle_started"] for row in rows),
        "retry_decision_cases": sum(bool(row["retry_decisions"]) for row in rows),
        "retry_allowed_cases": sum(row["retry_allowed"] for row in rows)}
    for key, expected in extra.items():
        need(metric[key] == expected, "Retry branch aggregate differs: " + key)
    reference.metric_check(easy, {k: v for k, v in metric.items() if k not in extra}, rows)


def verify_public(project):
    """Clone-safe checks retain failures and distinguish decisions from actions."""
    project = Path(project).resolve()
    base = load(project, "tools/adaptive_response_bundle.py", "retry20_clone_parent")
    reference = load(project, "tools/refine20_bundle.py", "retry20_clone_helpers")
    easy = load(project, "tools/easy20_bundle.py", "retry20_clone_metric_helpers")
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    need(selected.get("complete") is True and selected.get("kind") == KIND
         and selected.get("budget") == {**COUNTS, "total": 21}, "Complete21-case fixed retry report required")
    need(manifest.get("kind") == "retry20_fixed_strategy_research_publication_v1"
         and manifest["report_file"] == REPORT and manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["source_count"] == 28 and manifest["shared_source_count"] == 25
         and manifest["budget"] == selected["budget"], "Retry report/manifest differs")
    need(manifest["parent_manifests_unchanged"] is True
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes())
         and manifest["known_case_reference_report"] == OLD_REPORT
         and manifest["known_case_reference_report_sha256"] == digest((project / OLD_REPORT).read_bytes()),
         "Historical sources or the known failed-case reference changed")
    files, expected = manifest["files"], {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    roles = lambda role: {entry["original_file"] for entry in files if entry["role"] == role}
    need(len(files) == 14 and roles("runtime") == set(NEW_RUNTIME)
         and roles("posthoc_freeze_audit_or_test") == set(POSTHOC_NAMES)
         and roles("preregistered_plan_or_schedule") == {"experiment_plan.json", "pilot_matrix.json"}
         and roles("fixed_research_strategy") == {"fixed_strategy.json"}, "Retry explicit file allowlist changed")
    for entry in files:
        name = entry["published_file"]
        need(name.startswith(PUBLIC) and (project / name).resolve().is_relative_to(project / PUBLIC), "Retry archive path escaped")
        raw = (project / name).read_bytes()
        need(digest(raw) == entry["original_sha256"] == entry["published_sha256"]
             and entry["transform"] == "unchanged_raw_bytes", "Retry frozen bytes changed")
        if entry["role"] == "runtime":
            need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime source SHA differs")
        if entry["role"] == "posthoc_freeze_audit_or_test":
            need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed retry tool SHA differs")
        base.private_paths(raw)
        if name.endswith(".py"): compile(raw.decode("utf8"), name, "exec")
        expected.add(name)
    actual = {path.relative_to(project).as_posix() for path in (project / PUBLIC).rglob("*")
              if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc")}
    need(actual == expected and len(expected) == 17, "Extra SDK/log/cache/asset or missing retry archive file")
    bindings = manifest["shared_source_bindings"]
    need(len(bindings) == 25 and set(manifest["source_sha256"]) == set(NEW_RUNTIME) | {e["original_file"] for e in bindings}
         and not any("pour_policy_refine20/" in entry["original_file"] or entry["original_file"].endswith("prefix_control.py")
                     for entry in bindings), "Refine/prefix source became an undeclared runtime ancestor")
    for entry in bindings:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Shared V12 bytes changed")
    prior = read(project / OLD_REPORT)
    plan_path = project / PUBLIC / "experiment_plan.json"
    plan = read(plan_path)
    need(digest(plan_path.read_bytes()) == selected["experiment_plan_sha256"]
         and plan["draft"] is False and plan["budget"] == 21 and plan["budgets"] == COUNTS
         and plan["profiles"] == PROFILES and plan["test_seeds"] == sorted(SEEDS["test"])
         and plan["pilot_seed"] == 195301 and plan["pilot_is_seen_development_revisit"] is True
         and plan["data_fitting_or_model_selection"] is False,
         "Frozen fixed-strategy preregistration changed")
    need(digest((project / PUBLIC / "pilot_matrix.json").read_bytes())
         == selected["stage_artifact_sha256"]["pilot"]["matrix.json"], "Original pilot schedule changed")
    need(selected["task_contract"] == prior["task_contract"] and selected["physics"] == prior["physics"], "Task/physics declaration changed")
    need(selected["training_episodes"] == selected["validation_episodes"] == 0
         and selected["test_used_for_selection"] is False and selected["known_pilot_is_not_independent_test"] is True
         and selected["all_motion_speeds_fixed"] is True and selected["fixed_first_approach_tilt_deg"] == 84.
         and selected["fixed_retry_approach_tilt_deg"] == 84.1, "Fixed-rule study became training or test-based selection")
    rows = selected["cases"]
    need(len(rows) == 21 and len({row["episode_id"] for row in rows}) == 21, "All21 unique attempts required")
    for stage, count in COUNTS.items():
        group = [row for row in rows if row["stage"] == stage]
        need(len(group) == count and {row["seed"] for row in group} == SEEDS[stage], "Pilot/fresh budget or seeds changed")
        for role in {row["role"] for row in group}:
            metric_check(reference, easy, selected["stage_metrics"][stage][role], [row for row in group if row["role"] == role])
    for row in rows:
        amount, elapsed = row["received_ml"], row["physical_action_seconds"]
        need(all(type(row[key]) is bool for key in ("retry_enabled", "retry_allowed", "valid_evidence",
            "mass_conserved", "all_states_finite", "full_home_within_deadline", "no_late_dose_change",
            "refine_feasible", "easy_feasible", "original_strict_feasible", "original_strict_dose_passed"))
             and type(row["peak_all_label_spill_particles"]) is int and row["peak_all_label_spill_particles"] >= 0,
             "Typed physical/task flags or all-label spill count changed")
        need(row["target_ml"] == 20. and all(type(value) in (int, float) and math.isfinite(value)
             for value in (amount, elapsed, row["error_ml"], row["absolute_error_ml"], row["compute_wall_seconds_including_post"]))
             and abs(row["error_ml"] - (amount - 20.)) < 1e-9
             and abs(row["absolute_error_ml"] - abs(amount - 20.)) < 1e-9, "Actual volume/error changed")
        safety = bool(row["valid_evidence"] and row["mass_conserved"] and row["all_states_finite"]
            and row["full_home_within_deadline"] and elapsed <= 24.
            and row["peak_all_label_spill_particles"] <= 1 and row["no_late_dose_change"])
        need(row["refine_feasible"] is bool(safety and 19.5 <= amount <= 20.5)
             and row["easy_feasible"] is bool(safety and 19. <= amount <= 21.)
             and row["original_strict_dose_passed"] is (19.98 <= amount <= 20.02)
             and row["original_strict_feasible"] is bool(safety and 19.98 <= amount <= 20.02), "Three acceptance gates differ")
        need(row["split"] == ("development" if row["stage"] == "pilot" else "test")
             and row["known_previous_test_seed_used_as_development"] is (row["stage"] == "pilot")
             and row["retry_enabled"] is (row["role"] == "candidate"), "Seen pilot or paired retry mode relabelled")
        starts, ends, cycles = row["actual_correction_cycles_started"], row["actual_correction_cycles_ended"], row["actual_cycles"]
        need(0 <= ends <= starts <= (2 if row["retry_enabled"] else 1) and len(cycles) == starts
             and row["actual_second_cycle_started"] is (starts == 2), "Partial cycle/start/end counts differ")
        for index, cycle in enumerate(cycles):
            need(cycle["cycle_number"] == index + 1 and cycle["approach_tilt_deg"] == (84. if index == 0 else 84.1)
                 and cycle["has_recorded_endpoint"] is (index < ends), "Actual fixed angle sequence/endpoints differ")
            if cycle["has_recorded_endpoint"]:
                need(cycle["start_physical_s"] <= cycle["end_physical_s"] <= elapsed
                     and abs(cycle["outcome"]["settled_delta_ml"]
                             - (cycle["actual_end_received_ml"] - cycle["actual_start_received_ml"])) < 1e-8
                     and cycle["micro_skipped"] is (not cycle["executed_micro"]), "Measured cycle or skipped micro differs")
        decisions = row["retry_decisions"]
        need(len(decisions) <= 1 and (row["retry_enabled"] or not decisions), "Baseline or retry decision budget changed")
        for decision in decisions:
            t, budget, peak = decision["physical_time_s"], decision["conservative_full_home_budget_s"], decision["observed_all_label_peak_spill_particles"]
            spill = type(peak) is int and 0 <= peak <= 1
            fits = type(budget) in (int, float) and math.isfinite(budget) and budget >= 0. and t + budget <= 24.
            allowed = bool(decision["actual_settled_measurement"] and decision["actual_received_ml"] < 19.5 and spill and fits)
            need(decision["retry_allowed"] is allowed and decision["spill_guard_passed"] is spill
                 and decision["full_home_budget_fits"] is fits and decision["completed_cycles_before_decision"] == 1
                 and decision["declared_retry_approach_tilt_deg"] == 84.1
                 and decision["deficit_exceeds_half_ml"] is (decision["actual_received_ml"] < 19.5)
                 and abs(decision["actual_deficit_ml"] - (20. - decision["actual_received_ml"])) < 1e-9
                 and abs(decision["remaining_deadline_physical_seconds"] - (24. - t)) < 1e-9
                 and decision["deadline_physical_seconds"] == 24.
                 and decision["decision_is_intent_not_execution"] is True
                 and decision["budget_is_robot_motion_not_predicted_water_dose"] is True
                 and decision["actual_execution_evidence_event_type"] == "response_cycle_start",
                 "Actual conditional retry guard changed")
        need(row["retry_allowed"] is bool(decisions and decisions[0]["retry_allowed"])
             and (starts < 2 or (decisions and decisions[0]["retry_allowed"]
                                and decisions[0]["physical_time_s"] <= cycles[1]["start_physical_s"])),
             "Retry intent was substituted for actual execution")
        need(abs(row["post_physical_seconds"] - 2.) < 1e-8 and row["compute_wall_seconds_including_post"] > 0, "Post/wall timing changed")
    pilot = next(row for row in rows if row["stage"] == "pilot")
    known = pilot["known_pilot_reference"]
    old = next(row for row in prior["cases"] if row["stage"] == "test" and row["role"] == "candidate" and row["seed"] == 195301)
    need(pilot["refine_feasible"] is True and pilot["actual_second_cycle_started"] is True
         and len(pilot["actual_cycles"]) == 2 and all(c["settled"] for c in pilot["actual_cycles"])
         and known["first_cycle_water_motion_partitions_exactly_equal"] is True
         and known["first_cycle_original_events_exactly_equal"] is True
         and known["all_completed_frame_substep_counts_exactly_equal"] is True
         and known["reference_artifact_sha256"] == sorted(plan["pilot_reference_sha256"].values())
         and abs(known["original_failed_received_ml"] - old["received_ml"]) < 1e-9
         and abs(pilot["actual_cycles"][0]["actual_end_received_ml"] - old["received_ml"]) < 1e-9,
         "Seen development pilot did not repair and preserve the original first cycle")
    counter = known["zero_time_inherited_last_substeps"]
    need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
         and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
         and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
         and type(counter["reason"]) is str and bool(counter["reason"]),
         "Inherited t0 diagnostic counter was confused with completed physical updates")
    strategy_path = project / PUBLIC / "models/fixed_strategy.json"
    strategy = read(strategy_path)
    need(digest(strategy_path.read_bytes()) == selected["fixed_strategy"]["sha256"]
         and strategy["kind"] == "fixed20_conditional_second_cycle_strategy_v1" and strategy["task_kind"] == TASK_KIND
         and strategy["profiles"] == PROFILES and strategy["fixed_parameters"] == FIXED_PARAMETERS
         and strategy["source_sha256"] == manifest["source_sha256"]
         and strategy["input_sha256"] == manifest["input_sha256"]
         and strategy["gui_sha256"] == manifest["gui_default_sha256_unchanged"]
         and strategy["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
         and strategy["pilot_journal_sha256"] == selected["stage_artifact_sha256"]["pilot"]["episodes.jsonl"]
         and strategy["pilot_audit_sha256"] == selected["stage_artifact_sha256"]["pilot"]["audit.json"]
         and strategy["freeze_helper_sha256"] == REVIEWED_TOOLS[HERE + "freeze_strategy.py"]
         and strategy["pilot_uses_seen_previous_test_seed"] is True and strategy["pilot_joint_task_passed"] is True
         and strategy["no_data_fitting_or_model_selection"] is True and strategy["test_used_for_selection"] is False
         and strategy["default_gui_deployment"] is False, "Fixed strategy/freeze/provenance differs")
    need(all(type(value) in (int, float) for value in strategy["fixed_parameters"].values())
         and all(set(profile) == {"enable_retry"} and type(profile["enable_retry"]) is bool
                 for profile in strategy["profiles"].values()), "Fixed strategy has invalid parameter types")
    need(reference.timestamp(pilot["completed_at_utc"]) < reference.timestamp(strategy["created_at_utc"]), "Strategy froze before pilot gate")
    for row in rows:
        if row["stage"] == "test":
            need(reference.timestamp(strategy["created_at_utc"]) < reference.timestamp(row["started_at_utc"]), "Fresh tests preceded freeze")
    pairs, lookup = selected["test_pairs"], {row["episode_id"]: row for row in rows}
    need(len(pairs) == 10 and {pair["seed"] for pair in pairs} == SEEDS["test"], "Ten fresh independent pairs required")
    for pair in pairs:
        for role in PROFILES:
            need(pair[role] == lookup[pair[role]["episode_id"]] and pair[role]["role"] == role
                 and pair[role]["stage"] == "test" and pair[role]["seed"] == pair["seed"], "Paired fresh result changed")
    equivalence = selected["paired_first_cycle_equivalence"]
    need(len(equivalence) == 10 and {item["seed"] for item in equivalence} == SEEDS["test"],
         "All ten audited first-cycle comparisons required")
    pair_lookup = {pair["seed"]: pair for pair in pairs}
    for item in equivalence:
        pair = pair_lookup[item["seed"]]
        if item["endpoint_unavailable"]:
            need(item["verified"] is False and item["all_completed_frame_substep_counts_exactly_equal"] is None
                 and any(not pair[role]["actual_cycles"] or not pair[role]["actual_cycles"][0]["has_recorded_endpoint"]
                         for role in PROFILES), "Unavailable equivalence was relabelled as proof")
            continue
        need(item["verified"] is bool(item["original_first_cycle_events_exactly_equal"]
             and item["actual_water_motion_partitions_exactly_equal"])
             and item["all_completed_frame_substep_counts_exactly_equal"] is
                 (True if item["actual_water_motion_partitions_exactly_equal"] else None)
             and item["completed_frame_substep_equality_basis"] == "included_in_independent_physical_equality_conjunction",
             "Audited equivalence/substep conjunction changed")
        for role in PROFILES:
            first = pair[role]["actual_cycles"][0]
            need(item[role + "_episode_id"] == pair[role]["episode_id"] and first["has_recorded_endpoint"] is True
                 and abs(item[role + "_actual_first_cycle_end_received_ml"] - first["actual_end_received_ml"]) < 1e-9,
                 "Pair equivalence lost its actual endpoint/identity")
            if item["verified"]:
                need(abs(item["first_cycle_end_physical_s"] - first["end_physical_s"]) < 1e-9,
                     "Verified paired first-cycle time changed")
        counter = item["zero_time_inherited_last_substeps"]
        need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
             and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
             and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
             and counter["explanation_uses_same_audited_frozen_solver"] is True
             and type(counter["reason"]) is str and bool(counter["reason"]),
             "Paired inherited t0 diagnostic counter confused with actual completed updates")
    need(selected["all_paired_first_cycle_physical_states_equal"] is all(item["verified"] for item in equivalence),
         "First-cycle equivalence coverage claim differs")
    actual_retry_seeds = [row["seed"] for row in rows if row["stage"] == "test" and row["role"] == "candidate" and row["actual_second_cycle_started"]]
    need(selected["fresh_test_seeds_that_used_second_cycle"] == actual_retry_seeds, "Branch coverage was relabelled")
    need(selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and manifest["default_gui_deployment"] is False and manifest["all_continuous_goals_guaranteed"] is False
         and manifest["no_training_validation_or_selection"] is True
         and manifest["pilot_is_seen_development_not_independent"] is True
         and manifest["raw_logs_or_assets_distributed"] is False, "Unsupported training/default/range claim")
    need(len(manifest["plots"]) == 1, "Exactly one real retry figure required")
    for image in manifest["plots"]:
        raw = (project / image["file"]).read_bytes()
        need(image["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == image["sha256"], "Published real figure changed")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
