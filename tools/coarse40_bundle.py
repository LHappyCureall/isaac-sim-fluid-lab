"""Append an audited coarse-speed40mL study; preserve all old275 targets."""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

RUN = "droplet_out/pour_policy_coarse40/coarse40_001"
HERE = "projects/pour_policy_coarse40/"
PUBLIC = "research/coarse40_20261128/"
REPORT = "results/coarse40_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/coarse40_20261128/coarse40_results.png"
KIND = "coarse40_upstream_speed_calibration_v1"
COUNTS = {"calibration": 6, "validation": 4, "test": 20}
ORDER = ("coarse_252", "coarse_248")
SPEEDS = {"baseline": 25.6, "coarse_252": 25.2, "coarse_248": 24.8}
KNOWN_SEEDS = (197304, 198304)
NEW_RUNTIME = tuple(HERE + name for name in ("coarse_control.py", "run_coarse_test.py", "task_score.py"))
POSTHOC_NAMES = tuple(HERE + name for name in ("select_and_freeze.py", "audit_coarse_run.py",
    "audit_coarse_experiment.py", "curate_coarse_results.py", "plot_coarse_results.py", "test_coarse_control.py",
    "test_select_and_freeze.py", "test_audit_coarse_run.py", "test_curate_coarse_results.py"))
REVIEWED_TOOLS = {HERE + name: value for name, value in {
    "select_and_freeze.py": "794da7c030b8ae2d2ab62e4039df1cba36ebf8ce8ee8548fc641e4bdc7f0ab86",
    "audit_coarse_run.py": "f5f9b277f828d7ad913115965c6f8e4e71ddbdc2481a4f0c9481d56ca74e82b4",
    "audit_coarse_experiment.py": "ca8987d753478641037e8ac4c10e195de5fb4cb619a62704e2c19e49307bffe8",
    "curate_coarse_results.py": "4076894aaa1ab81a9a2f486754f986b6e26d8e17df2c785620213cd4ca433e80",
    "plot_coarse_results.py": "247d44d3e621d5840c87a9101ae7ce923320e363b0027b73b58dd74f3db2f514",
    "test_coarse_control.py": "160a5202b1e8373a9ee64183aec1d5ed95ba63f82e64398d91f221af858ed92a",
    "test_select_and_freeze.py": "6a63d861963b7a937b0358d82cd37a690b9219c07252bfd9adad2d7c163a5458",
    "test_audit_coarse_run.py": "ea1946c6581ad4806b6d76ce5f223f056d5e366da7f7aa09e50c318daa3f4516",
    "test_curate_coarse_results.py": "d4ee4b9267606226800f64379c29dc57c63ccd73fe6b823ae6dba3dde11de63b",
}.items()}
ATTRIBUTES = ("# Preserve frozen runtime, profiles, checkpoints and source provenance.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n")
README = ("# Upstream coarse-speed-only 40 mL calibration\n\n"
    "Only coarse dosing speed varies: baseline 25.6 degrees/s and candidates 25.2/24.8 degrees/s. Both "
    "roles keep first approach 87 degrees, retry 87.2 degrees, conditional retry enabled and at most two "
    "cycles. All other coarse/micro speeds, holds, geometry, particles and solver parameters are fixed. "
    "See [the actual selected results](../../results/coarse40_20261128.json).\n\n"
    "Six seen calibration cases include two conditions for each profile. A baseline may fail dose, "
    "spill, home or late-water task gates and remain a valid negative comparator only when its physical, "
    "mass and measurement evidence is valid. No such relaxation applies to candidate acceptance. Both "
    "candidate cases must satisfy every primary criterion. Feasible groups rank only mean raw physical "
    "action time; exact ties use 25.2 then 24.8 degrees/s. Other valid groups rank mean absolute error then "
    "total all-label peak spill. Invalid groups cannot be elites. This is finite calibration, not CEM, "
    "PPO or a trained neural policy.\n\n"
    "One choice freezes before two fresh validation pairs. Validation can authorize twenty new paired "
    "tests or stop; it cannot retune or choose a fallback. All 6, 10 or 30 actual cases and failures remain. "
    "A stopped study includes an explicit receipt and no invented checkpoint or test outcomes.\n\n"
    "Primary dose is 40 +/-0.5 mL; +/-1 mL and original +/-0.1% (0.04 mL) remain separate. Full return "
    "to original pregrasp with 12 quiet frames must finish within 24 raw physical seconds, peak spill "
    "across all labels must be at most one particle and another two seconds must show no late dose "
    "change. All required numerical, mass, measurement and actual-motion checks also apply.\n\n"
    "Two known baselines are bound to prior first40 chosen-angle traces, including the valid spill-two "
    "negative example. Equality across speeds is checked only for preparation before coarse dosing; "
    "the later coarse-anchor and whole correction cycles are expected to change. The inherited "
    "time-zero diagnostic substep counter is distinct from every checked completed solver frame.\n\n"
    "Three runtime files reference 25 unchanged V12 dependencies. Fifty protected old 20/30/first40 "
    "bindings refer to existing public files without copying them again. Original and portable SHA "
    "values remain distinct, including the existing robot path conversion. Posthoc audit dependencies "
    "retain their V12/V13 relative paths and pinned V9 reference. Reconstruct licensed resources and "
    "the original module layout under a new protocol; never rebind guards to resume private journals.\n\n"
    "No SDK, third-party robot assets, NPZ, raw step logs, video or personal paths are included. "
    "Kinematic pregrasp and particle-mass truth are not a real scale, contact-grasp or hardware test. "
    "A fixed 40 mL task and narrow randomization do not guarantee arbitrary 20--40 mL or replace a GUI default.\n")


def need(value, message):
    if not value: raise ValueError(message)


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
    normalize = lambda value: json.loads(json.dumps({k: v for k, v in value.items() if k not in ignored}, allow_nan=False))
    # Pure CPU group keys are tuples; their frozen JSON representation is an array.
    return normalize(left) == normalize(right)


def frozen_json_bytes(value, checkpoint_raw):
    """Reproduce the frozen selector's text-writer newline convention."""
    crlf, lf = checkpoint_raw.count(b"\r\n"), checkpoint_raw.count(b"\n")
    need(lf > 0 and crlf in (0, lf), "Frozen checkpoint has an unsupported mixed newline convention")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    return raw.replace(b"\n", b"\r\n") if crlf else raw


def gate(workspace, project):
    # No held-out journals or checkpoints are inspected before final independent evidence exists.
    audit_path = workspace / RUN / "overall_audit.json"
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("kind") == KIND,
         "A complete independent coarse40 experiment audit is required")
    stages = tuple(audit["stage_artifact_sha256"])
    need(stages in (("calibration",), ("calibration", "validation"), tuple(COUNTS)),
         "Only actual6/10-case stops or the complete30-case study may publish")
    for stage in stages:
        summary, job = read(workspace / RUN / stage / "summary.json"), read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == COUNTS[stage]
             and job.get("state") == "complete" and job.get("pid") is None
             and not (workspace / RUN / stage / "worker.lock").exists(), "Every actual stage must finish")
    if "test" not in stages:
        need(not (workspace / RUN / "test").exists(), "Stopped study cannot hide actual test attempts")
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "Coarse40 posthoc allowlist differs")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64 and digest((workspace / name).read_bytes()) == expected,
             "Final reviewed coarse40 tool required: " + name)
    need(audit["auditor_source_file"] == HERE + "audit_coarse_experiment.py"
         and audit["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_coarse_experiment.py"],
         "Final independent coarse40 auditor changed")
    base = load(project, "tools/adaptive_response_bundle.py", "coarse40_publication_parent")
    curator = load(workspace, HERE + "curate_coarse_results.py", "coarse40_reviewed_curator")
    selected = read(project / REPORT)
    need(same_summary(selected, curator.build_summary(workspace / RUN, audit_path, digest(audit_path.read_bytes()))),
         "Selected coarse40 report differs from every actual audited attempt")
    base.private_paths((project / REPORT).read_bytes())
    return base, selected, stages


def protected_bindings(protected, prior, project):
    """Resolve old original hashes to existing raw or reviewed portable public bytes."""
    prior = dict(prior)
    # Selected reports were manually curated and are outside the source sync map.
    for relative in ("results/retry20_20261128.json", "results/retry3040_20261128.json", "results/first40_20261128.json"):
        need("projects/isaac-sim-fluid-lab/" + relative in protected, "All three old selected reports must remain protected")
        raw = (project / relative).read_bytes()
        need(digest(raw) == protected["projects/isaac-sim-fluid-lab/" + relative], "An old selected report changed")
        prior[relative] = raw
    manifests = []
    for name, raw in prior.items():
        if name.endswith("source_manifest.json"):
            data = json.loads(raw)
            manifests.append((name, digest(raw), data.get("files", []) + data.get("shared_source_bindings", [])))
    output = []
    for original, expected in sorted(protected.items()):
        direct = original.removeprefix("projects/isaac-sim-fluid-lab/")
        if direct in prior and digest(prior[direct]) == expected:
            output.append({"original_file": original, "original_sha256": expected, "published_file": direct,
                "published_sha256": expected, "transform": "existing_raw_public_bytes", "source_manifest_file": None})
            continue
        matches = [(name, sha, entry) for name, sha, entries in manifests for entry in entries
                   if entry.get("original_file") == original and entry.get("original_sha256") == expected
                   and entry.get("published_file") in prior
                   and digest(prior[entry["published_file"]]) == entry["published_sha256"]]
        if matches:
            name, sha, entry = sorted(matches, key=lambda x: (x[0], x[2]["published_file"]))[0]
            output.append({"original_file": original, "original_sha256": expected,
                "published_file": entry["published_file"], "published_sha256": entry["published_sha256"],
                "transform": entry["transform"], "source_manifest_file": name, "source_manifest_sha256": sha})
            continue
        identical = sorted(name for name, raw in prior.items() if digest(raw) == expected)
        need(identical, "Protected old version has no existing reviewed public binding: " + original)
        output.append({"original_file": original, "original_sha256": expected, "published_file": identical[0],
            "published_sha256": expected, "transform": "existing_raw_public_bytes", "source_manifest_file": None})
    need(len(output) == len({r["original_file"] for r in output}) == 50, "Exactly50 old protected version references required")
    return output


def verify_protected(project, manifest):
    protected, bindings = manifest["protected_file_sha256"], manifest["protected_file_bindings"]
    need(len(protected) == len(bindings) == 50 and {r["original_file"] for r in bindings} == set(protected),
         "Old20/30 protected-version reference inventory changed")
    for entry in bindings:
        need(entry["original_sha256"] == protected[entry["original_file"]]
             and digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"],
             "An old protected public version changed")
        if entry["source_manifest_file"] is None:
            need(entry["original_sha256"] == entry["published_sha256"] and entry["transform"] == "existing_raw_public_bytes",
                 "Raw protected-version provenance differs")
        else:
            source = project / entry["source_manifest_file"]
            need(digest(source.read_bytes()) == entry["source_manifest_sha256"], "An old source manifest changed")
            data = read(source)
            candidates = data.get("files", []) + data.get("shared_source_bindings", [])
            need(any(all(candidate.get(key) == entry[key] for key in ("original_file", "original_sha256", "published_file",
                     "published_sha256", "transform")) for candidate in candidates), "Original/portable old-source SHA was rebound")


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base, selected, stages = gate(workspace, project)
    prior = dict(prior_prepared)
    parent_raw = prior[base.PUBLIC + "source_manifest.json"]
    entries = {e["original_file"]: e for e in json.loads(parent_raw)["files"] if e["role"] == "collection_runtime"}
    shared = set(entries) - {"projects/pour_policy_adaptive_learning/collect_cycles.py"}
    protocols = {stage: read(workspace / RUN / stage / "protocol.json") for stage in stages}
    calibration = protocols["calibration"]
    sources, protected = calibration["source_sha256"], calibration["protected_file_sha256"]
    need(len(shared) == 25 and len(sources) == 28 and set(sources) == shared | set(NEW_RUNTIME), "Exact25 shared/3 new runtime inventory required")
    bindings = []
    for name in sorted(shared):
        entry = entries[name]
        need(sources[name] == entry["original_sha256"] and digest(prior[entry["published_file"]]) == entry["published_sha256"],
             "Shared V12 source changed")
        bindings.append(entry)
    for protocol in protocols.values():
        need(all(protocol[key] == calibration[key] for key in ("source_sha256", "input_sha256", "gui_sha256", "protected_file_sha256"))
             and len(protocol["input_sha256"]) == 8 and len(protocol["protected_file_sha256"]) == 50,
             "Sources/static inputs/old versions changed across phases")
    old_bindings = protected_bindings(protected, prior, project)
    files, mapping = [], []

    def add(original, path, role, expected, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected, "Explicit coarse40 source/checkpoint changed: " + original)
        base.private_paths(raw)
        destination = destination or PUBLIC + original
        if destination.endswith(".py"): ast.parse(raw.decode("utf8"), filename=destination)
        files.append((destination, raw))
        mapping.append({"role": role, "original_file": original, "published_file": destination,
            "original_sha256": expected, "published_sha256": digest(raw), "transform": "unchanged_raw_bytes"})

    for name in NEW_RUNTIME: add(name, workspace / RUN / "calibration/source_snapshot" / name, "runtime", sources[name])
    for name, expected in REVIEWED_TOOLS.items(): add(name, workspace / name, "posthoc_selection_audit_or_test", expected)
    metadata = [("experiment_plan.json", workspace / RUN / "experiment_plan.json"),
        ("calibration_matrix.json", workspace / RUN / "calibration/matrix.json"),
        ("profiles_calibration.json", workspace / RUN / "calibration/input_snapshot" / calibration["matrix"]["profile_file"]),
        ("protected_versions.json", workspace / RUN / "calibration/input_snapshot" / read(workspace / RUN / "experiment_plan.json")["protected_versions_file"]),
        ("fresh_seed_scan.json", workspace / HERE / "fresh_seed_scan.json")]
    for name, path in metadata: add(name, path, "plan_profile_protected_or_seed_metadata", digest(path.read_bytes()), PUBLIC + name)
    selection = selected["calibration_selection"]
    if selection is not None:
        add(selection["file"], base.inside(workspace / RUN, selection["file"]), "pre_validation_calibration_selection",
            selection["sha256"], PUBLIC + "models/calibration_selection.json")
    final = selected["fixed_strategy"] if "test" in stages else selected["finalization"]
    destination = PUBLIC + ("models/fixed_strategy.json" if "test" in stages else "development_stop_receipt.json")
    add(final["file"], base.inside(workspace / RUN, final["file"]),
        "pre_test_fixed_strategy" if "test" in stages else "explicit_stopped_development_receipt", final["sha256"], destination)
    plot = read(workspace / PLOTS / "plot_provenance.json")
    plot_report = workspace / PLOTS / "coarse40_results.json"
    image = (workspace / PLOTS / "coarse40_results.png").read_bytes()
    need(plot["complete"] is True and plot["synthetic_fixture"] is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_coarse_results.py"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_coarse_results.py"]
         and same_summary(read(plot_report), selected)
         and digest(plot_report.read_bytes()) == plot["files_sha256"]["coarse40_results.json"]
         and image.startswith(b"\x89PNG\r\n\x1a\n") and digest(image) == plot["files_sha256"]["coarse40_results.png"],
         "Actual coarse40 figure/input provenance changed")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "coarse40_single_factor_publication_v1", "report_file": REPORT,
        "report_sha256": digest((project / REPORT).read_bytes()), "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "experiment_plan_sha256": selected["experiment_plan_sha256"], "actual_stages": list(stages), "budget": selected["budget"],
        "source_count": 28, "shared_source_count": 25, "source_sha256": sources, "shared_source_bindings": bindings,
        "input_sha256": calibration["input_sha256"], "gui_default_sha256_unchanged": calibration["gui_sha256"],
        "protected_file_sha256": protected, "protected_file_bindings": old_bindings,
        "protected_manifest_raw_sha256": calibration["protected_manifest_raw_sha256"],
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mapping,
        "parent_adaptive_manifest_sha256": digest(parent_raw), "parent_manifests_unchanged": True,
        "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}], "only_coarse_dose_speed_varied": True,
        "both_roles_retry_enabled": True, "fixed_first_approach_tilt_deg": 87., "fixed_retry_approach_tilt_deg": 87.2,
        "bounded_calibration_performed": True, "validation_selected_or_retuned_candidate": False,
        "baseline_comparator_does_not_require_task_success": True, "candidate_group_requires_all_joint_primary": True,
        "test_used_for_profile_selection": False, "original_strict_standard_unchanged": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False, "raw_logs_or_assets_distributed": False}
    raw = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw)
    files.extend(((PUBLIC + "source_manifest.json", raw), (PUBLIC + "README.md", README.encode()), (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    expected_count = 22 if selection is None else 23
    need(len(files) == len({name for name, _ in files}) == expected_count and not {name for name, _ in files} & set(prior),
         "Only the explicit22/23 new coarse40 targets may be appended")
    return files


def case_check(row, profile, profile_input):
    goal, amount, elapsed = row["target_ml"], row["received_ml"], row["physical_action_seconds"]
    flags = ("valid_evidence", "mass_conserved", "all_states_finite", "full_home_within_deadline",
        "no_late_dose_change", "refine_feasible", "easy_feasible", "original_strict_feasible",
        "original_strict_dose_passed", "retry_enabled", "actual_second_cycle_started", "retry_allowed")
    need(all(type(row[key]) is bool for key in flags)
         and type(row["peak_all_label_spill_particles"]) is int and row["peak_all_label_spill_particles"] >= 0,
         "Task/physics flags and actual all-label spill count must be explicit")
    need(goal == 40. and all(type(v) in (int, float) and math.isfinite(v) for v in (amount, elapsed, row["error_ml"], row["absolute_error_ml"]))
         and abs(row["error_ml"] - (amount - goal)) < 1e-9 and abs(row["absolute_error_ml"] - abs(amount - goal)) < 1e-9, "Actual3040 volume/error changed")
    safe = bool(row["valid_evidence"] and row["mass_conserved"] and row["all_states_finite"] and row["full_home_within_deadline"]
        and elapsed <= 24. and row["peak_all_label_spill_particles"] <= 1 and row["no_late_dose_change"])
    need(row["refine_feasible"] is bool(safe and goal - .5 <= amount <= goal + .5)
         and row["easy_feasible"] is bool(safe and goal - 1. <= amount <= goal + 1.)
         and row["original_strict_dose_passed"] is (goal - goal * .001 <= amount <= goal + goal * .001)
         and row["original_strict_feasible"] is bool(safe and goal - goal * .001 <= amount <= goal + goal * .001), "Three target-specific closed gates differ")
    expected = profile
    need(row["retry_enabled"] is True and row["profile_namespace"] == "40"
         and row["profile_sha256"] == profile_input["sha256"]
         and row["first_approach_tilt_deg"] == expected["first_approach_tilt_deg"] and row["retry_approach_tilt_deg"] == expected["retry_approach_tilt_deg"], "Executed goal profile differs")
    starts, ends, cycles = row["actual_correction_cycles_started"], row["actual_correction_cycles_ended"], row["actual_cycles"]
    need(type(starts) is int and type(ends) is int and 0 <= ends <= starts <= (2 if row["retry_enabled"] else 1) and len(cycles) == starts
         and row["actual_second_cycle_started"] is (starts == 2), "Actual/partial cycle counts differ")
    for index, cycle in enumerate(cycles):
        need(cycle["cycle_number"] == index + 1 and cycle["approach_tilt_deg"] == expected["first_approach_tilt_deg" if index == 0 else "retry_approach_tilt_deg"]
             and cycle["has_recorded_endpoint"] is (index < ends), "Actual per-goal cycle angles/endpoints differ")
        if cycle["has_recorded_endpoint"]:
            need(cycle["start_physical_s"] <= cycle["end_physical_s"] <= elapsed
                 and abs(cycle["outcome"]["settled_delta_ml"] - (cycle["actual_end_received_ml"] - cycle["actual_start_received_ml"])) < 1e-8
                 and abs(cycle["outcome"]["duration_s"] - (cycle["end_physical_s"] - cycle["start_physical_s"])) < 1e-8
                 and type(cycle["settled"]) is bool and type(cycle["executed_micro"]) is bool
                 and cycle["micro_skipped"] is (not cycle["executed_micro"])
                 and cycle["action_execution_complete"] is bool(cycle["settled"] and cycle["executed_micro"])
                 and cycle["completion_event_type"] == ("response_cycle_complete" if cycle["settled"] else "response_cycle_unsettled"),
                 "Measured cycle/approach drainage/skipped micro differs")
    decisions = row["retry_decisions"]
    need(len(decisions) <= 1 and (row["retry_enabled"] or not decisions), "Retry decision budget changed")
    for d in decisions:
        t, budget, peak = d["physical_time_s"], d["conservative_full_home_budget_s"], d["observed_all_label_peak_spill_particles"]
        spill = type(peak) is int and 0 <= peak <= 1
        fits = type(budget) in (int, float) and math.isfinite(budget) and budget >= 0. and t + budget <= 24.
        allowed = bool(d["actual_settled_measurement"] and d["actual_received_ml"] < goal - .5 and spill and fits)
        need(d["declared_target_ml"] == goal and d["target_profile_sha256"] == row["profile_sha256"]
             and d["condition_seed"] == row["coarse40_condition_seed"] == row["seed"]
             and d["target_profile_id"] == row["profile_id"] and d["declared_retry_approach_tilt_deg"] == expected["retry_approach_tilt_deg"]
             and d["completed_cycles_before_decision"] == 1 and d["retry_allowed"] is allowed and d["spill_guard_passed"] is spill
             and d["target_profile_key"] == row["profile_namespace"] and d["deficit_exceeds_half_ml"] is (d["actual_received_ml"] < goal - .5)
             and d["full_home_budget_fits"] is fits and abs(d["actual_deficit_ml"] - (goal - d["actual_received_ml"])) < 1e-9
             and d["deadline_physical_seconds"] == 24. and abs(d["remaining_deadline_physical_seconds"] - (24. - t)) < 1e-9
             and d["decision_is_intent_not_execution"] is True and d["budget_is_robot_motion_not_predicted_water_dose"] is True, "Actual goal-relative retry guards changed")
    need(row["retry_allowed"] is bool(decisions and decisions[0]["retry_allowed"])
         and (starts < 2 or (decisions and decisions[0]["retry_allowed"] and decisions[0]["physical_time_s"] <= cycles[1]["start_physical_s"])), "Intent was counted as executed retry")
    need(row["compute_wall_seconds_including_post"] > 0 and row["action_compute_wall_seconds"] > 0,
         "Actual compute timings must remain positive")
    if row["valid_evidence"]:
        need(row["post_observation_available"] is True and row["post_observation_valid_complete"] is True
             and row["post_logged_frames"] == 120 and row["post_physical_steps_reported"] == 120
             and abs(row["post_physical_seconds"] - 2.) < 1e-8, "Valid evidence requires all120 actual post frames")
    else:
        need(row["refine_feasible"] is False and row["easy_feasible"] is False and row["original_strict_feasible"] is False,
             "Partial or invalid evidence cannot pass a task criterion")


def baseline_valid(row):
    return bool(row["valid_evidence"] and row["valid_physics"] is True
                and row["mass_conserved"] and row["all_states_finite"])


def group_score(rows):
    need(len(rows) == 2 and {r["seed"] for r in rows} == set(KNOWN_SEEDS), "Every group needs both seen cases")
    if any(not r["valid_evidence"] for r in rows): return None
    if all(r["refine_feasible"] for r in rows): return [0, sum(r["physical_action_seconds"] for r in rows) / 2.]
    return [1, sum(r["absolute_error_ml"] for r in rows) / 2., sum(r["peak_all_label_spill_particles"] for r in rows)]


def compact_selection_case(row):
    return {"episode_id": row["episode_id"], "profile_key": row["profile_key"], "role": row["role"],
        "seed": row["seed"], "coarse_dose_tilt_deg_s": row["coarse_dose_tilt_deg_s"],
        "actual_received_ml": row["received_ml"], "error_ml": row["error_ml"], "absolute_error_ml": row["absolute_error_ml"],
        "physical_time_s": row["physical_action_seconds"], "all_label_peak_spill_particles": row["peak_all_label_spill_particles"],
        "valid_evidence": row["valid_evidence"], "primary_passed": row["refine_feasible"]}


def phase_proof_check(proof, stage, selected, reference):
    rows = [r for r in selected["cases"] if r["stage"] == stage]
    need(proof["stage"] == stage and proof["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_coarse_run.py"]
         and {**proof["input_artifact_sha256"], "audit.json": proof["audit_sha256"]} == selected["stage_artifact_sha256"][stage]
         and proof["actual_cases"] == [compact_selection_case(r) for r in rows]
         and set(proof["completed_sidecar_sha256"]) == {r["episode_id"] for r in rows}
         and reference.timestamp(proof["created_at_utc"]) > max(reference.timestamp(r["completed_at_utc"]) for r in rows),
         "Frozen completed-stage proof lost actual cases or changed audit bindings")
    need(all("traces/" + r["episode_id"] + ".steps.jsonl.gz" in proof["raw_artifact_sha256"] for r in rows),
         "Actual action trace SHA disappeared")
    need(all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
             for value in (*proof["completed_sidecar_sha256"].values(), *proof["raw_artifact_sha256"].values())),
         "Original evidence needs explicit raw SHA256 bindings")


def known_proof_check(project, selected, manifest, plan):
    proofs, rows = selected["known_baseline_equivalence"], selected["cases"]
    need(set(proofs) == {str(seed) for seed in KNOWN_SEEDS}, "Both known positive/negative reference proofs are required")
    old = read(project / "results/first40_20261128.json")
    for seed in KNOWN_SEEDS:
        proof = proofs[str(seed)]
        original = next(r for r in old["cases"] if r["seed"] == seed and r["role"] == "candidate"
                        and r["stage"] == ("calibration" if seed == 197304 else "test"))
        current = next(r for r in rows if r["stage"] == "calibration" and r["role"] == "baseline" and r["seed"] == seed)
        need(proof["seen_development_case_not_independent"] is True
             and proof["reference_artifact_sha256"] == sorted(plan["known_reference_sha256"][str(seed)].values()),
             "Known baseline reference identity was relabelled")
        unavailable = proof.get("endpoint_unavailable") is True and proof["verified"] is False
        fields = (("original_reference_received_ml", "received_ml"),
                  ("original_reference_primary_feasible", "refine_feasible"),
                  ("original_reference_peak_spill_particles", "peak_all_label_spill_particles"))
        need(all((unavailable and proof[name] is None) or proof[name] == original[field] for name, field in fields),
             "Known reference values were invented or a negative task outcome was hidden")
        if not proof["verified"]:
            need(selected["calibration_selection"] is None, "Unverified known physics cannot authorize validation")
            continue
        need(proof["all_completed_frame_substep_counts_exactly_equal"] is True
             and proof["first_cycle_water_motion_partitions_exactly_equal"] is True
             and proof["first_cycle_original_events_exactly_equal"] is True
             and proof["actual_water_motion_partitions_exactly_equal"] is True
             and current["received_ml"] == original["received_ml"]
             and current["actual_cycles"] and current["actual_cycles"][0]["has_recorded_endpoint"] is True,
             "Verified known first-cycle physics differs")
        endpoint = current["actual_cycles"][0]["end_physical_s"]
        need(abs(proof["first_cycle_end_physical_s"] - endpoint) < 1e-9
             and abs(proof["first_cycle_frames"] - (endpoint * 60. + 1.)) < 1e-8, "Known first-cycle comparison endpoint changed")
        counter = proof["zero_time_inherited_last_substeps"]
        need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
             and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
             and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
             and min(counter["reference"], counter["pilot"]) >= 4 and bool(counter["reason"]),
             "Inherited t0 diagnostics cannot hide completed physical steps")


def prepare_proof_check(selected, manifest, chosen):
    stages, rows, proofs = selected["actual_stages"], selected["cases"], selected["prepare_prefix_equivalence"]
    expected = {("calibration", seed, key) for seed in KNOWN_SEEDS for key in ORDER}
    expected.update((stage, seed, chosen) for stage in stages if stage != "calibration"
                    for seed in ((198201, 198202) if stage == "validation" else range(198401, 198411)))
    need(len(proofs) == len(expected) and {(p["stage"], p["seed"], p["candidate_id"]) for p in proofs} == expected,
         "All actual prepare-only control comparisons must remain")
    for proof in proofs:
        group = {r["candidate_id"]: r for r in rows if r["stage"] == proof["stage"] and r["seed"] == proof["seed"]}
        need(proof["baseline_episode_id"] == group["baseline"]["episode_id"]
             and proof["candidate_episode_id"] == group[proof["candidate_id"]]["episode_id"]
             and type(proof["verified"]) is bool, "Prepare-only proof lost actual condition identities")
        if proof.get("endpoint_unavailable", False):
            need(proof["verified"] is False, "Unavailable prepare endpoint cannot prove equality")
            continue
        endpoint, times = proof["prepare_endpoint_physical_s"], proof["maximum_compared_dense_time_s"]
        need(proof["comparison_endpoint_kind"] == "actual_prepared78_before_coarse_dose_speed_treatment"
             and proof["no_future_kinematics_included"] is True
             and proof["post_coarse_anchor_equality_is_not_required"] is True
             and proof["whole_first_cycle_equality_is_not_required"] is True
             and proof["verified"] is proof["actual_prepare_water_motion_partitions_exactly_equal"]
             and proof["all_completed_frame_substep_counts_exactly_equal"] is (True if proof["verified"] else None)
             and type(endpoint) in (int, float) and math.isfinite(endpoint) and endpoint >= 0.
             and type(proof["prepare_prefix_frames"]) is int and proof["prepare_prefix_frames"] > 0
             and isinstance(times, list) and len(times) == 2
             and all(t is None or (type(t) in (int, float) and math.isfinite(t) and 0. <= t <= endpoint) for t in times),
             "Prepare proof included or erased treatment effects")
        counter = proof["zero_time_inherited_last_substeps"]
        need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
             and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
             and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
             and type(counter["baseline"]) is int and type(counter["candidate"]) is int and bool(counter["reason"]),
             "Inherited prepare t0 counter was confused with actual completed updates")
    need(selected["all_prepare_prefix_physical_states_equal"] is all(p["verified"] for p in proofs),
         "Prepare-prefix evidence coverage differs")


def verify_public(project):
    """Verify every actual negative/positive case; never require baseline task success."""
    project = Path(project).resolve()
    base = load(project, "tools/adaptive_response_bundle.py", "coarse40_clone_parent")
    retry = load(project, "tools/retry20_bundle.py", "coarse40_clone_metrics")
    retry3040 = load(project, "tools/retry3040_bundle.py", "coarse40_clone_water_rules")
    refine = load(project, "tools/refine20_bundle.py", "coarse40_clone_reference")
    easy = load(project, "tools/easy20_bundle.py", "coarse40_clone_easy")
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    stages, stopped = selected["actual_stages"], selected["stopped_before_independent_test"]
    need(selected["complete"] is True and selected["kind"] == KIND and type(stopped) is bool
         and stages in (["calibration"], ["calibration", "validation"], list(COUNTS))
         and stopped is ("test" not in stages) and selected["independent_test_not_run"] is stopped
         and selected["budget"] == {stage: COUNTS[stage] if stage in stages else 0 for stage in COUNTS}
         and selected["actual_cases"] == sum(COUNTS[s] for s in stages) and selected["maximum_actual_cases"] == 30
         and selected["outcome"] == ("fresh_test_completed" if not stopped else
             "validation_gate_failed" if "validation" in stages else "calibration_gate_failed"),
         "Only actual6/10-case stops or the complete30-case study may publish")
    need(manifest["kind"] == "coarse40_single_factor_publication_v1" and manifest["report_file"] == REPORT
         and manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and manifest["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
         and manifest["stage_artifact_sha256"] == selected["stage_artifact_sha256"]
         and manifest["actual_stages"] == stages and manifest["budget"] == selected["budget"]
         and manifest["source_count"] == 28 and manifest["shared_source_count"] == 25
         and manifest["parent_manifests_unchanged"] is True
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes()),
         "New/old report-source provenance differs")
    has_selection = selected["calibration_selection"] is not None
    need(has_selection is ("validation" in stages), "A calibration stop cannot invent a chosen checkpoint")
    mapping, expected = manifest["files"], {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    by_original = {entry["original_file"]: entry for entry in mapping}
    names = lambda role: {entry["original_file"] for entry in mapping if entry["role"] == role}
    need(len(mapping) == len(by_original) == (19 if has_selection else 18)
         and names("runtime") == set(NEW_RUNTIME) and names("posthoc_selection_audit_or_test") == set(POSTHOC_NAMES)
         and names("plan_profile_protected_or_seed_metadata") == {"experiment_plan.json", "calibration_matrix.json",
             "profiles_calibration.json", "protected_versions.json", "fresh_seed_scan.json"}
         and names("pre_validation_calibration_selection") == ({"calibration_selection.json"} if has_selection else set())
         and names("explicit_stopped_development_receipt" if stopped else "pre_test_fixed_strategy")
             == {selected["finalization" if stopped else "fixed_strategy"]["file"]}, "Explicit22/23-file allowlist differs")
    for entry in mapping:
        name = entry["published_file"]
        need(name.startswith(PUBLIC) and (project / name).resolve().is_relative_to(project / PUBLIC), "Archive target escaped")
        raw = (project / name).read_bytes()
        need(digest(raw) == entry["original_sha256"] == entry["published_sha256"] and entry["transform"] == "unchanged_raw_bytes",
             "Frozen original coarse40 bytes changed")
        if entry["role"] == "runtime": need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime binding changed")
        if entry["role"] == "posthoc_selection_audit_or_test": need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed posthoc source changed")
        base.private_paths(raw)
        if name.endswith(".py"): compile(raw.decode("utf8"), name, "exec")
        expected.add(name)
    actual = {p.relative_to(project).as_posix() for p in (project / PUBLIC).rglob("*") if p.is_file()
              and not (p.parent.name == "__pycache__" and p.suffix == ".pyc")}
    need(actual == expected and len(expected) == (22 if has_selection else 21), "Extra SDK/cache/log or missing archive file")
    bindings = manifest["shared_source_bindings"]
    need(len(bindings) == len({entry["original_file"] for entry in bindings}) == 25
         and len(manifest["source_sha256"]) == 28 and len(manifest["input_sha256"]) == 8
         and len(manifest["gui_default_sha256_unchanged"]) == 3
         and set(manifest["source_sha256"]) == set(NEW_RUNTIME) | {entry["original_file"] for entry in bindings}
         and not any("/pour_policy_first40/" in entry["original_file"] or "/pour_policy_retry20/" in entry["original_file"]
                     or "/pour_policy_retry3040/" in entry["original_file"] for entry in bindings), "Old controllers became runtime ancestors")
    for entry in bindings:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Shared V12 dependency changed")
    for name, sha in manifest["gui_default_sha256_unchanged"].items():
        need(digest((project / name).read_bytes()) == sha, "An old GUI/default changed")
    verify_protected(project, manifest)
    plan = read(project / PUBLIC / "experiment_plan.json")
    profiles = read(project / PUBLIC / "profiles_calibration.json")
    protected = read(project / PUBLIC / "protected_versions.json")
    need(digest((project / PUBLIC / "experiment_plan.json").read_bytes()) == selected["experiment_plan_sha256"]
         and plan["draft"] is False and plan["target_ml"] == 40. and plan["maximum_actual_cases"] == 30
         and plan["stages"]["calibration"]["seeds"] == list(KNOWN_SEEDS)
         and plan["stages"]["calibration"]["candidate_speeds_in_tie_order_deg_s"] == [25.2, 24.8]
         and plan["stages"]["validation"]["seeds"] == [198201, 198202]
         and plan["stages"]["test"]["seeds"] == list(range(198401, 198411)), "Grouped plan or fresh conditions changed")
    need(profiles["profiles"] == {key: {"enable_retry": True, "first_approach_tilt_deg": 87.,
        "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": speed} for key, speed in SPEEDS.items()}
         and digest((project / PUBLIC / "calibration_matrix.json").read_bytes()) == selected["stage_artifact_sha256"]["calibration"]["matrix.json"],
         "Static speed profiles or original calibration matrix changed")
    need(selected["task_contract"] == {"target_ml": 40., **plan["task"], "secondary_easy_absolute_tolerance_ml": 1.,
        "original_strict_absolute_tolerance_ml": .04}
         and selected["physics"] == {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
        "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
        "total_particles": 11625, "source_particles": 2880,
        "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "Primary/original strict task or measurement meaning changed")
    need(protected["files_sha256"] == selected["protected_file_sha256"] == manifest["protected_file_sha256"]
         and digest((project / PUBLIC / "protected_versions.json").read_bytes()) == plan["protected_versions_raw_sha256"]
         == selected["protected_manifest_raw_sha256"] == manifest["protected_manifest_raw_sha256"]
         and digest((project / PUBLIC / "fresh_seed_scan.json").read_bytes()) == plan["fresh_seed_scan_sha256"], "Protected50/fresh-scan bindings changed")
    rows = selected["cases"]
    need(len(rows) == len({row["episode_id"] for row in rows}) == selected["actual_cases"], "Every actual physical attempt must remain")
    calibration = [r for r in rows if r["stage"] == "calibration"]
    need(len(calibration) == 6 and {(r["candidate_id"], r["seed"]) for r in calibration}
         == {(key, seed) for key in SPEEDS for seed in KNOWN_SEEDS}, "Keep all six seen cases including negative baseline")
    scores = {key: group_score([r for r in calibration if r["candidate_id"] == key]) for key in ORDER}
    options = [(tuple(score), index, key) for index, key in enumerate(ORDER) if (score := scores[key]) is not None]
    chosen = min(options)[2] if options else None
    need(selected["calibration_selected_candidate_id_recomputed"] == chosen
         and selected["calibration_candidate_group_keys"] == scores, "Grouped mean-time/MAE/spill calibration ranking changed")
    for row in rows:
        stage, key = row["stage"], row["candidate_id"]
        need(stage in stages and row["profile_key"] == key and key in SPEEDS
             and row["role"] == ("baseline" if key == "baseline" else "candidate")
             and row["profile_namespace"] == "40" and row["retry_enabled"] is True
             and row["profile_id"] == profiles["profile_id"] and row["coarse_dose_tilt_deg_s"] == SPEEDS[key]
             and row["coarse40_condition_seed"] == row["seed"] and row["condition_seed_provided_by_runner"] is True
             and row["split"] == ("test" if stage == "test" else "development")
             and row["known_seen_development_case"] is (stage == "calibration")
             and row["validation_is_not_final_holdout"] is (stage == "validation")
             and (stage == "calibration" or key == ("baseline" if row["role"] == "baseline" else chosen)),
             "Treatment speed, identical retry policy, seed provenance or phase labels changed")
        case_check(row, profiles["profiles"][key], {"sha256": digest((project / PUBLIC / "profiles_calibration.json").read_bytes())})
        need(row["second_cycle_execution_or_skip_reason"] == retry3040.retry_reason(row), "Actual retry/skip outcome changed")
    for stage in stages:
        group = [r for r in rows if r["stage"] == stage]
        if stage != "calibration":
            seeds = (198201, 198202) if stage == "validation" else range(198401, 198411)
            need(len(group) == COUNTS[stage] and {(r["role"], r["seed"]) for r in group}
                 == {(role, seed) for role in ("baseline", "candidate") for seed in seeds}, "Complete actual paired grid required")
        for role in ("baseline", "candidate"):
            retry.metric_check(refine, easy, selected["stage_metrics"][stage][role], [r for r in group if r["role"] == role])
    for key in SPEEDS:
        retry.metric_check(refine, easy, selected["calibration_metrics_by_candidate"][key], [r for r in calibration if r["candidate_id"] == key])
    lookup = {r["episode_id"]: r for r in rows}
    need(set(selected["paired_comparisons"]) == set(stages) - {"calibration"}, "Actual paired phase coverage changed")
    for stage, pairs in selected["paired_comparisons"].items():
        seeds = (198201, 198202) if stage == "validation" else tuple(range(198401, 198411))
        need(len(pairs) == len(seeds) and {p["seed"] for p in pairs} == set(seeds), "All actual paired conditions required")
        for pair in pairs:
            for role in ("baseline", "candidate"):
                need(pair[role] == lookup[pair[role]["episode_id"]] and pair[role]["role"] == role
                     and pair[role]["stage"] == stage and pair[role]["seed"] == pair["seed"], "Paired result lost an actual case")
            baseline, candidate = pair["baseline"], pair["candidate"]
            need(abs(pair["candidate_minus_baseline_absolute_error_ml"] - (candidate["absolute_error_ml"] - baseline["absolute_error_ml"])) < 1e-9
                 and abs(pair["candidate_minus_baseline_physical_seconds"] - (candidate["physical_action_seconds"] - baseline["physical_action_seconds"])) < 1e-9
                 and pair["candidate_minus_baseline_peak_spill_particles"] == candidate["peak_all_label_spill_particles"] - baseline["peak_all_label_spill_particles"],
                 "Actual paired accuracy/time/spill comparison changed")
    need(selected["test_pairs"] == selected["paired_comparisons"].get("test", []), "Tests were removed or invented")
    selector = load(project, by_original[HERE + "select_and_freeze.py"]["published_file"], "coarse40_clone_selector")
    curator = load(project, by_original[HERE + "curate_coarse_results.py"]["published_file"], "coarse40_clone_compact")
    models = []
    if has_selection:
        path = project / PUBLIC / "models/calibration_selection.json"
        selection = read(path);models.append(selection)
        need(digest(path.read_bytes()) == selected["calibration_selection"]["sha256"]
             and selection["kind"] == "coarse40_calibration_selection_v1" and selection["calibration_gate_passed"] is True
             and selection["selected_profile_key"] == chosen and scores[chosen] is not None and scores[chosen][0] == 0
             and selection["candidate_tie_order"] == list(ORDER) and selection["candidate_scores"] == scores
             and selection["all_calibration_cases"] == [compact_selection_case(r) for r in calibration]
             and selection["validation_used_for_selection"] is False and selection["test_used_for_selection"] is False
             and selection["created_at_utc"] == selected["calibration_selection"]["created_at_utc"]
             and all(baseline_valid(r) for r in calibration if r["role"] == "baseline"), "Frozen grouped choice or valid-negative-baseline gate changed")
        ranks = [((2, index) if scores[key] is None else (*scores[key], index), key) for index, key in enumerate(ORDER)]
        need(selection["candidate_ranking"] == [{"profile_key": key, "rank": list(rank)} for rank, key in sorted(ranks)], "Exact mean-time tie order changed")
        phase_proof_check(selection["calibration_proof"], "calibration", selected, refine)
        need(refine.timestamp(selection["created_at_utc"]) > refine.timestamp(selection["calibration_proof"]["created_at_utc"])
             and all(refine.timestamp(selection["created_at_utc"]) < refine.timestamp(r["started_at_utc"]) for r in rows if r["stage"] == "validation"),
             "Choice did not freeze after calibration audit and before validation")
        matrix = selector.make_matrix("validation", selection, digest(path.read_bytes()))
        need(digest(frozen_json_bytes(matrix, path.read_bytes())) == selected["stage_artifact_sha256"]["validation"]["matrix.json"], "Validation matrix lost its frozen choice")
    if stopped:
        path = project / PUBLIC / "development_stop_receipt.json";receipt = read(path)
        need(selected["fixed_strategy"] is None and selected["test_pairs"] == []
             and digest(path.read_bytes()) == selected["finalization"]["sha256"]
             and receipt["kind"] == "coarse40_finalization_v1" and receipt["outcome"] == selected["outcome"]
             and receipt["actual_stages"] == stages and receipt["independent_test_cases"] == 0
             and receipt["no_retune_or_baseline_fallback"] is True, "Stopped calibration invented a checkpoint or tests")
    else:
        path = project / PUBLIC / "models/fixed_strategy.json";strategy = read(path);models.append(strategy)
        need(digest(path.read_bytes()) == selected["fixed_strategy"]["sha256"]
             and strategy["kind"] == "coarse40_frozen_speed_strategy_v1" and strategy["validation_gate_passed"] is True
             and strategy["calibration_selection_sha256"] == selected["calibration_selection"]["sha256"]
             and strategy["candidate_not_retuned_or_replaced_by_validation"] is True
             and strategy["created_at_utc"] == selected["fixed_strategy"]["created_at_utc"]
             and all(baseline_valid(r) for r in rows if r["stage"] == "validation" and r["role"] == "baseline")
             and all(r["refine_feasible"] for r in rows if r["stage"] == "validation" and r["role"] == "candidate"),
             "Fresh validation did not authorize the unchanged group with valid negative controls")
        phase_proof_check(strategy["validation_proof"], "validation", selected, refine)
        need(refine.timestamp(strategy["created_at_utc"]) > refine.timestamp(strategy["validation_proof"]["created_at_utc"])
             and all(refine.timestamp(strategy["created_at_utc"]) < refine.timestamp(r["started_at_utc"]) for r in rows if r["stage"] == "test"),
             "Final strategy did not freeze before all actual test resets")
        matrix = selector.make_matrix("test", selection, selected["calibration_selection"]["sha256"], digest(path.read_bytes()))
        need(digest(frozen_json_bytes(matrix, path.read_bytes())) == selected["stage_artifact_sha256"]["test"]["matrix.json"], "Test matrix retuned or lost its frozen strategy")
    for model in models:
        need(model["task_kind"] == "fixed40ml_absolute05ml_24s_onegrain_v1" and model["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
             and model["profiles"] == profiles["profiles"] and model["selected_profile_key"] == chosen
             and model["selected_coarse_dose_tilt_deg_s"] == SPEEDS[chosen] and model["first_approach_tilt_deg"] == 87.
             and model["retry_approach_tilt_deg"] == 87.2 and model["baseline_coarse_dose_tilt_deg_s"] == 25.6
             and model["both_roles_enable_retry"] is True and model["maximum_cycles"] == 2
             and model["only_coarse_dose_speed_varies"] is True and model["fixed_parameters"] == selector.FIXED_PARAMETERS
             and model["source_sha256"] == manifest["source_sha256"] and model["input_sha256"] == manifest["input_sha256"]
             and model["gui_sha256"] == manifest["gui_default_sha256_unchanged"] and model["protected_file_sha256"] == manifest["protected_file_sha256"]
             and model["selector_source_sha256"] == REVIEWED_TOOLS[HERE + "select_and_freeze.py"]
             and model["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_coarse_run.py"]
             and model["test_used_for_selection"] is False and model["default_gui_deployment"] is False,
             "A frozen model changed more than the selected coarse dosing speed")
        need({key: curator.compact_known_reference(value) for key, value in model["known_baseline_equivalence"].items()} == selected["known_baseline_equivalence"]
             and model["profile_raw_sha256"] == digest((project / PUBLIC / "profiles_calibration.json").read_bytes())
             and model["input_sha256"][model["profile_file"]] == model["profile_raw_sha256"]
             and model["input_sha256"][plan["protected_versions_file"]] == plan["protected_versions_raw_sha256"], "Known-reference/static profile provenance changed")
    known_proof_check(project, selected, manifest, plan)
    prepare_proof_check(selected, manifest, chosen)
    need(selected["old20_and30_versions_unchanged"] is True and selected["first40_versions_unchanged"] is True
         and selected["bounded_coarse_speed_calibration"] is True
         and selected["feasible_calibration_rank_uses_only_mean_physical_time_then_declared_order"] is True
         and selected["infeasible_calibration_rank_uses_mean_absolute_error_then_total_peak_spill"] is True
         and selected["baseline_comparator_does_not_require_task_success"] is True and selected["candidate_group_requires_all_joint_primary"] is True
         and selected["selected_from_calibration_not_validation_or_test"] is True
         and selected["collector_no_data_fit_or_model_selection_header_scope"] == "collector_only_not_the_finite_calibration_experiment"
         and selected["only_coarse_dose_speed_varies"] is True and selected["only_first_angle_varies"] is False
         and selected["first_angle_fixed_deg"] == 87. and selected["retry_angle_fixed_deg"] == 87.2 and selected["both_roles_enable_retry"] is True
         and selected["post_coarse_anchor_equality_across_changed_speeds_is_not_expected"] is True
         and selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False,
         "Unsupported task relaxation, causal boundary, calibration or GUI/default claim")
    need(manifest["only_coarse_dose_speed_varied"] is True and manifest["both_roles_retry_enabled"] is True
         and manifest["fixed_first_approach_tilt_deg"] == 87. and manifest["fixed_retry_approach_tilt_deg"] == 87.2
         and manifest["baseline_comparator_does_not_require_task_success"] is True and manifest["candidate_group_requires_all_joint_primary"] is True
         and manifest["test_used_for_profile_selection"] is False and manifest["validation_selected_or_retuned_candidate"] is False
         and manifest["default_gui_deployment"] is False and manifest["raw_logs_or_assets_distributed"] is False, "Publication scope claim differs")
    need(len(manifest["plots"]) == 1, "One actual final figure required")
    for image in manifest["plots"]:
        raw = (project / image["file"]).read_bytes()
        need(image["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == image["sha256"], "Actual selected figure changed")
    base.private_paths((project / REPORT).read_bytes());base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
