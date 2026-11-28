"""Append one completed first-angle40mL study; preserve the old252 targets."""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

RUN = "droplet_out/pour_policy_first40/first40_001"
HERE = "projects/pour_policy_first40/"
PUBLIC = "research/first40_20261128/"
REPORT = "results/first40_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/first40_20261128/first40_results.png"
KIND = "first40_first_angle_calibration_v1"
COUNTS = {"calibration": 4, "validation": 4, "test": 20}
ORDER = ("first_8700", "first_8705", "first_8695")
ANGLES = {"baseline": 87.1, "first_8700": 87., "first_8705": 87.05, "first_8695": 86.95}
NEW_RUNTIME = tuple(HERE + name for name in ("first_control.py", "run_first_test.py", "task_score.py"))
POSTHOC_NAMES = tuple(HERE + name for name in ("select_and_freeze.py", "audit_first_run.py",
    "audit_first_experiment.py", "curate_first_results.py", "plot_first_results.py", "test_first_control.py",
    "test_select_and_freeze.py", "test_audit_first_run.py", "test_curate_first_results.py"))
REVIEWED_TOOLS = {HERE + name: value for name, value in {
    "select_and_freeze.py": "694394edea738fc5fdd7dc42363a73c499906601187536712836a268a11ba9dc",
    "audit_first_run.py": "5ccd055256c5147430ff83db4c576dd17d32863ab1a4ad2cfc45e9aaf2ae654e",
    "audit_first_experiment.py": "a6b6a9bb6551b23b9ac46cb3bcf2fddb5edc50f5b490e95fc17b84c68df7d0d2",
    "curate_first_results.py": "83c127ac203f982a562a4d9e4fc1739648862a5bc67468596623ef5fb6b65bb8",
    "plot_first_results.py": "4e9ce66d446ee873466802b6924046fa32e8611f3c4795d14861e7985fc8dc46",
    "test_first_control.py": "bf4e460cc68a4640b4d65532b3054918965ff4c217c37e577644376f09f6d358",
    "test_select_and_freeze.py": "7aa7a0bb0932bf9044071bd01389d8e29e7eb6aa872a73d47c356b3cbb68cbbe",
    "test_audit_first_run.py": "d360447f1f6c426e489d4987323c569699111def05245dbfab64e41462d9514c",
    "test_curate_first_results.py": "1f012fc7bcbfd56134d99ce6e5e9ec910a5211b86ac526391de0b3cf8aa044c8",
}.items()}
ATTRIBUTES = ("# Preserve frozen runtime, profiles, checkpoints and source provenance.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n")
README = ("# First-angle-only 40 mL calibration\n\n"
    "This separate study changes only the first correction approach angle for 40 mL. Both roles keep "
    "conditional retry enabled, at most two cycles, the second approach fixed at 87.2 degrees and all "
    "coarse/micro speeds, holds, particles and physics unchanged. Baseline uses 87.1 degrees. This is "
    "not the earlier retry-disabled/retry-enabled comparison. See "
    "[the actual selected results](../../results/first40_20261128.json).\n\n"
    "The known failed seed 197304 is seen calibration data. Baseline and three candidate first angles "
    "87.0, 87.05 and 86.95 degrees are retained. Among primary-feasible candidates, only raw physical "
    "action duration is ranked; exact ties use that declared grid order. Invalid evidence cannot be an "
    "elite. One choice freezes before two fresh validation seeds. Validation can authorize the final "
    "twenty-case paired test, or stop the study; it cannot retune or choose a fallback. No test selects "
    "the angle. A finite grid is parameter calibration, not PPO, CEM or a learned neural policy.\n\n"
    "The primary band is 40 +/-0.5 mL. The +/-1 mL secondary and original +/-0.1% (0.04 mL) criterion "
    "remain separate. All require return to original pregrasp and 12 stable frames within 24 raw physical "
    "seconds, all-label peak spill at most one particle, valid mass/numerics/motion and another two "
    "seconds without late dose change. Post observation cannot rescue the action deadline.\n\n"
    "The known baseline is bound to the old enabled-retry first-cycle evidence, including actual fluid "
    "and motion equality and every completed-frame substep. The inherited diagnostic counter at time "
    "zero is documented separately. New first angles intentionally change the first correction cycle. "
    "Their common coarse prefix is checked separately; it is not a claim of equal full first cycles. "
    "Intent, actual second cycles, approach drainage and skipped micro-actions remain distinct.\n\n"
    "Three runtime files reference 25 unchanged V12 dependencies. Forty-two old version bindings resolve "
    "existing public source, profile, model, report and GUI bytes without copying them again. Original "
    "and portable public SHA values remain separate, including the prior robot path conversion. "
    "Old 20 and 30 mL implementations and results are unchanged. Posthoc audit dependencies retain "
    "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py, audit_response_collection.py "
    "and cycle_samples.py, plus projects/pour_policy_training_v9/audit_fine_run.py, at their original "
    "relative paths and literal SHA guards. Restore the licensed resources "
    "and original module layout under a new protocol; do not rebind SHA guards to resume private logs.\n\n"
    "No SDK, third-party robot assets, NPZ, raw step logs, video or personal paths are included. "
    "Particle mass truth and kinematic pregrasp are not a real scale, contact grasp or hardware test. "
    "This fixed 40 mL study does not guarantee arbitrary 20--40 mL or replace a GUI default.\n")


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
    return {k: v for k, v in left.items() if k not in ignored} == {k: v for k, v in right.items() if k not in ignored}


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
         "A complete independent first40 experiment audit is required")
    stages = tuple(audit["stage_artifact_sha256"])
    need(stages in (("calibration", "validation"), tuple(COUNTS)), "Only actual8-case stop or complete28-case study may publish")
    for stage in stages:
        summary, job = read(workspace / RUN / stage / "summary.json"), read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == COUNTS[stage]
             and job.get("state") == "complete" and job.get("pid") is None
             and not (workspace / RUN / stage / "worker.lock").exists(), "Every actual stage must finish")
    if "test" not in stages:
        need(not (workspace / RUN / "test").exists(), "Stopped study cannot hide actual test attempts")
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "First40 posthoc allowlist differs")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64 and digest((workspace / name).read_bytes()) == expected,
             "Final reviewed first40 tool required: " + name)
    need(audit["auditor_source_file"] == HERE + "audit_first_experiment.py"
         and audit["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_first_experiment.py"],
         "Final independent first40 auditor changed")
    base = load(project, "tools/adaptive_response_bundle.py", "first40_publication_parent")
    curator = load(workspace, HERE + "curate_first_results.py", "first40_reviewed_curator")
    selected = read(project / REPORT)
    need(same_summary(selected, curator.build_summary(workspace / RUN, audit_path, digest(audit_path.read_bytes()))),
         "Selected first40 report differs from every actual audited attempt")
    base.private_paths((project / REPORT).read_bytes())
    return base, selected, stages


def protected_bindings(protected, prior, project):
    """Resolve old original hashes to existing raw or reviewed portable public bytes."""
    prior = dict(prior)
    # Selected reports were manually curated and are outside the source sync map.
    for relative in ("results/retry20_20261128.json", "results/retry3040_20261128.json"):
        need("projects/isaac-sim-fluid-lab/" + relative in protected, "Both old selected reports must remain protected")
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
    need(len(output) == len({r["original_file"] for r in output}) == 42, "Exactly42 old protected version references required")
    return output


def verify_protected(project, manifest):
    protected, bindings = manifest["protected_file_sha256"], manifest["protected_file_bindings"]
    need(len(protected) == len(bindings) == 42 and {r["original_file"] for r in bindings} == set(protected),
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
             and len(protocol["input_sha256"]) == 8 and len(protocol["protected_file_sha256"]) == 42,
             "Sources/static inputs/old versions changed across phases")
    old_bindings = protected_bindings(protected, prior, project)
    files, mapping = [], []

    def add(original, path, role, expected, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected, "Explicit first40 source/checkpoint changed: " + original)
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
    add(selection["file"], base.inside(workspace / RUN, selection["file"]), "pre_validation_calibration_selection",
        selection["sha256"], PUBLIC + "models/calibration_selection.json")
    final = selected["fixed_strategy"] if "test" in stages else selected["finalization"]
    destination = PUBLIC + ("models/fixed_strategy.json" if "test" in stages else "development_stop_receipt.json")
    add(final["file"], base.inside(workspace / RUN, final["file"]),
        "pre_test_fixed_strategy" if "test" in stages else "explicit_stopped_development_receipt", final["sha256"], destination)
    plot = read(workspace / PLOTS / "plot_provenance.json")
    plot_report = workspace / PLOTS / "first40_results.json"
    image = (workspace / PLOTS / "first40_results.png").read_bytes()
    need(plot["complete"] is True and plot["synthetic_fixture"] is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_first_results.py"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_first_results.py"]
         and same_summary(read(plot_report), selected)
         and digest(plot_report.read_bytes()) == plot["files_sha256"]["first40_results.json"]
         and image.startswith(b"\x89PNG\r\n\x1a\n") and digest(image) == plot["files_sha256"]["first40_results.png"],
         "Actual first40 figure/input provenance changed")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "first40_single_factor_publication_v1", "report_file": REPORT,
        "report_sha256": digest((project / REPORT).read_bytes()), "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "experiment_plan_sha256": selected["experiment_plan_sha256"], "actual_stages": list(stages), "budget": selected["budget"],
        "source_count": 28, "shared_source_count": 25, "source_sha256": sources, "shared_source_bindings": bindings,
        "input_sha256": calibration["input_sha256"], "gui_default_sha256_unchanged": calibration["gui_sha256"],
        "protected_file_sha256": protected, "protected_file_bindings": old_bindings,
        "protected_manifest_raw_sha256": calibration["protected_manifest_raw_sha256"],
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mapping,
        "parent_adaptive_manifest_sha256": digest(parent_raw), "parent_manifests_unchanged": True,
        "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}], "only_first_angle_varied": True,
        "both_roles_retry_enabled": True, "fixed_retry_approach_tilt_deg": 87.2,
        "bounded_calibration_performed": True, "validation_selected_or_retuned_candidate": False,
        "test_used_for_profile_selection": False, "original_strict_standard_unchanged": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False, "raw_logs_or_assets_distributed": False}
    raw = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw)
    files.extend(((PUBLIC + "source_manifest.json", raw), (PUBLIC + "README.md", README.encode()), (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len(files) == len({name for name, _ in files}) == 23 and not {name for name, _ in files} & set(prior),
         "Only23 new first40 targets may be appended")
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
             and d["condition_seed"] == row["first40_condition_seed"] == row["seed"]
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


def prefix_proof_check(selected, manifest, chosen):
    """Compare the common coarse prefix; treatment cycles intentionally differ."""
    stages, rows, records = selected["actual_stages"], selected["cases"], selected["coarse_prefix_equivalence"]
    expected = {("calibration", 197304, key) for key in ORDER}
    expected.update((stage, seed, chosen) for stage in stages if stage != "calibration"
                    for seed in ((198101, 198102) if stage == "validation" else range(198301, 198311)))
    need(len(records) == len(expected) and {(r["stage"], r["seed"], r["candidate_id"]) for r in records} == expected,
         "Every actual coarse-prefix control comparison must be retained")
    for record in records:
        group = {row["candidate_id"]: row for row in rows if row["stage"] == record["stage"] and row["seed"] == record["seed"]}
        baseline, candidate = group["baseline"], group[record["candidate_id"]]
        need(record["baseline_episode_id"] == baseline["episode_id"] and record["candidate_episode_id"] == candidate["episode_id"]
             and type(record["verified"]) is bool, "Coarse-prefix proof lost actual condition identities")
        if record.get("endpoint_unavailable", False):
            need(record["verified"] is False and (not baseline["first_quiet_anchor_available"] or not candidate["first_quiet_anchor_available"]),
                 "Unavailable coarse endpoints cannot be relabelled as equal physics")
            continue
        need(record["comparison_endpoint_kind"] == "actual_quiet_coarse78_before_first_angle_treatment"
             and record["whole_first_cycle_equality_is_not_required"] is True
             and record["verified"] is record["actual_coarse_water_motion_partitions_exactly_equal"]
             and record["all_completed_frame_substep_counts_exactly_equal"] is (True if record["verified"] else None),
             "Coarse-prefix proof was confused with changed-angle whole-cycle equality")
        for role, row in (("baseline", baseline), ("candidate", candidate)):
            need(record[role + "_actual_coarse_anchor_received_ml"] == row["first_quiet_anchor_received_ml"]
                 and record[role + "_actual_coarse_anchor_physical_s"] == row["first_quiet_anchor_physical_seconds"],
                 "Actual coarse-prefix endpoint was replaced")
            if record["verified"]:
                need(row["first_quiet_anchor_available"] is True
                     and record["coarse_anchor_physical_s"] == row["first_quiet_anchor_physical_seconds"],
                     "Verified common-prefix clock differs")
        if record["verified"]:
            need(baseline["first_quiet_anchor_received_ml"] == candidate["first_quiet_anchor_received_ml"],
                 "Verified common-prefix water differs")
        counter = record["zero_time_inherited_last_substeps"]
        need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
             and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
             and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
             and type(counter["baseline"]) is int and type(counter["candidate"]) is int
             and bool(counter["reason"]), "Inherited t0 diagnostics cannot hide completed fluid updates")
    need(selected["all_coarse_prefix_physical_states_equal"] is all(r["verified"] for r in records),
         "Common-prefix evidence coverage claim differs")


def known_proof_check(selected, manifest, plan):
    proof = selected["known_baseline_equivalence"]
    baseline = next(row for row in selected["cases"] if row["stage"] == "calibration" and row["candidate_id"] == "baseline")
    need(proof["verified"] is True and proof["seen_development_case_not_independent"] is True
         and proof["all_completed_frame_substep_counts_exactly_equal"] is True
         and proof["first_cycle_water_motion_partitions_exactly_equal"] is True
         and proof["first_cycle_original_events_exactly_equal"] is True
         and proof["actual_water_motion_partitions_exactly_equal"] is True
         and proof["reference_artifact_sha256"] == sorted(plan["known_reference_sha256"].values())
         and proof["original_failed_received_ml"] == baseline["received_ml"],
         "Known old enabled-retry failure was relabelled or lost its actual equivalence proof")
    need(baseline["actual_cycles"] and baseline["actual_cycles"][0]["has_recorded_endpoint"] is True,
         "Known first-cycle endpoint is missing")
    first = baseline["actual_cycles"][0]
    need(abs(proof["first_cycle_end_physical_s"] - first["end_physical_s"]) < 1e-9
         and abs(proof["first_cycle_frames"] - (first["end_physical_s"] * 60. + 1.)) < 1e-8,
         "Known reference comparison stopped at a different physical endpoint")
    counter = proof["zero_time_inherited_last_substeps"]
    need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
         and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
         and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
         and type(counter["reference"]) is int and type(counter["pilot"]) is int
         and counter["reference"] >= 4 and counter["pilot"] >= 4 and bool(counter["reason"]),
         "Known-reference t0 diagnostic exception was confused with actual completed solver steps")


def compact_selection_case(row):
    return {"episode_id": row["episode_id"], "profile_key": row["profile_key"], "role": row["role"],
        "seed": row["seed"], "actual_received_ml": row["received_ml"], "error_ml": row["error_ml"],
        "absolute_error_ml": row["absolute_error_ml"], "physical_time_s": row["physical_action_seconds"],
        "all_label_peak_spill_particles": row["peak_all_label_spill_particles"],
        "valid_evidence": row["valid_evidence"], "primary_passed": row["refine_feasible"]}


def phase_proof_check(proof, stage, selected, reference):
    rows = [row for row in selected["cases"] if row["stage"] == stage]
    need(proof["stage"] == stage and proof["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_first_run.py"]
         and {**proof["input_artifact_sha256"], "audit.json": proof["audit_sha256"]} == selected["stage_artifact_sha256"][stage]
         and proof["actual_cases"] == [compact_selection_case(row) for row in rows]
         and set(proof["completed_sidecar_sha256"]) == {row["episode_id"] for row in rows}
         and reference.timestamp(proof["created_at_utc"]) > max(reference.timestamp(row["completed_at_utc"]) for row in rows),
         "Frozen completed-stage proof discarded actual cases or changed the audit/sidecar bindings")
    for row in rows:
        need("traces/" + row["episode_id"] + ".steps.jsonl.gz" in proof["raw_artifact_sha256"],
             "Actual action trace SHA disappeared from frozen phase proof")
    for value in (*proof["completed_sidecar_sha256"].values(), *proof["raw_artifact_sha256"].values()):
        need(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
             "Original evidence must retain explicit raw SHA256 bindings")


def verify_public(project):
    """Clone-safe bounded-study checks; negative test outcomes remain publishable."""
    project = Path(project).resolve()
    base = load(project, "tools/adaptive_response_bundle.py", "first40_clone_parent")
    retry = load(project, "tools/retry20_bundle.py", "first40_clone_metrics")
    retry3040 = load(project, "tools/retry3040_bundle.py", "first40_clone_water_rules")
    refine = load(project, "tools/refine20_bundle.py", "first40_clone_reference")
    easy = load(project, "tools/easy20_bundle.py", "first40_clone_easy")
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    stages = selected["actual_stages"]
    stopped = selected["stopped_before_independent_test"]
    need(selected["complete"] is True and selected["kind"] == KIND and type(stopped) is bool
         and stages in (["calibration", "validation"], list(COUNTS)) and stopped is ("test" not in stages)
         and selected["independent_test_not_run"] is stopped
         and selected["budget"] == {"calibration": 4, "validation": 4, "test": 0 if stopped else 20}
         and selected["actual_cases"] == (8 if stopped else 28) and selected["maximum_actual_cases"] == 28
         and selected["outcome"] == ("validation_gate_failed" if stopped else "fresh_test_completed"),
         "Complete actual8-case stop or28-case first40 study required")
    need(manifest["kind"] == "first40_single_factor_publication_v1" and manifest["report_file"] == REPORT
         and manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and manifest["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
         and manifest["stage_artifact_sha256"] == selected["stage_artifact_sha256"]
         and manifest["actual_stages"] == stages and manifest["budget"] == selected["budget"]
         and manifest["source_count"] == 28 and manifest["shared_source_count"] == 25
         and manifest["parent_manifests_unchanged"] is True
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes()),
         "New/old report and source provenance changed")
    mapping = manifest["files"]
    by_original = {entry["original_file"]: entry for entry in mapping}
    expected = {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    names = lambda role: {entry["original_file"] for entry in mapping if entry["role"] == role}
    metadata = {"experiment_plan.json", "calibration_matrix.json", "profiles_calibration.json", "protected_versions.json", "fresh_seed_scan.json"}
    need(len(mapping) == len(by_original) == 19 and names("runtime") == set(NEW_RUNTIME)
         and names("posthoc_selection_audit_or_test") == set(POSTHOC_NAMES)
         and names("plan_profile_protected_or_seed_metadata") == metadata
         and names("pre_validation_calibration_selection") == {"calibration_selection.json"}
         and names("explicit_stopped_development_receipt" if stopped else "pre_test_fixed_strategy")
             == {selected["finalization" if stopped else "fixed_strategy"]["file"]}, "Explicit first40 allowlist differs")
    for entry in mapping:
        name = entry["published_file"]
        need(name.startswith(PUBLIC) and (project / name).resolve().is_relative_to(project / PUBLIC), "Archive target escaped")
        raw = (project / name).read_bytes()
        need(digest(raw) == entry["original_sha256"] == entry["published_sha256"] and entry["transform"] == "unchanged_raw_bytes",
             "Original frozen first40 bytes changed")
        if entry["role"] == "runtime": need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime binding differs")
        if entry["role"] == "posthoc_selection_audit_or_test": need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed posthoc source changed")
        base.private_paths(raw)
        if name.endswith(".py"): compile(raw.decode("utf8"), name, "exec")
        expected.add(name)
    actual = {p.relative_to(project).as_posix() for p in (project / PUBLIC).rglob("*") if p.is_file()
              and not (p.parent.name == "__pycache__" and p.suffix == ".pyc")}
    need(actual == expected and len(expected) == 22, "Extra SDK/cache/log or missing first40 archive file")
    bindings = manifest["shared_source_bindings"]
    need(len(bindings) == len({entry["original_file"] for entry in bindings}) == 25
         and len(manifest["source_sha256"]) == 28
         and set(manifest["source_sha256"]) == set(NEW_RUNTIME) | {entry["original_file"] for entry in bindings}
         and not any("/pour_policy_retry20/" in entry["original_file"] or "/pour_policy_retry3040/" in entry["original_file"] for entry in bindings),
         "New runtime unexpectedly imports an old20/30 controller")
    for entry in bindings:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Shared V12 dependency changed")
    verify_protected(project, manifest)
    plan = read(project / PUBLIC / "experiment_plan.json")
    profiles = read(project / PUBLIC / "profiles_calibration.json")
    protected = read(project / PUBLIC / "protected_versions.json")
    need(digest((project / PUBLIC / "experiment_plan.json").read_bytes()) == selected["experiment_plan_sha256"]
         and plan["draft"] is False and plan["target_ml"] == 40. and plan["maximum_actual_cases"] == 28
         and plan["stages"]["calibration"]["candidate_angles_in_tie_order"] == [87., 87.05, 86.95]
         and plan["stages"]["validation"]["seeds"] == [198101, 198102]
         and plan["stages"]["test"]["seeds"] == list(range(198301, 198311)), "Bounded plan/candidate order/fresh seeds changed")
    need(profiles["profiles"] == {key: {"enable_retry": True, "first_approach_tilt_deg": angle,
        "retry_approach_tilt_deg": 87.2} for key, angle in ANGLES.items()}, "Static four-profile input changed")
    need(digest((project / PUBLIC / "calibration_matrix.json").read_bytes()) == selected["stage_artifact_sha256"]["calibration"]["matrix.json"]
         and len(manifest["input_sha256"]) == 8 and len(manifest["gui_default_sha256_unchanged"]) == 3,
         "Initial calibration matrix or static input/GUI inventory changed")
    for name, expected_sha in manifest["gui_default_sha256_unchanged"].items():
        need(digest((project / name).read_bytes()) == expected_sha, "An old public GUI/default changed")
    need(selected["task_contract"] == {"target_ml": 40., **plan["task"], "secondary_easy_absolute_tolerance_ml": 1.,
        "original_strict_absolute_tolerance_ml": .04}
         and selected["physics"] == {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
        "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
        "total_particles": 11625, "source_particles": 2880,
        "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "Primary/original strict task or simulation measurement meaning changed")
    need(protected["files_sha256"] == selected["protected_file_sha256"] == manifest["protected_file_sha256"]
         and digest((project / PUBLIC / "protected_versions.json").read_bytes()) == plan["protected_versions_raw_sha256"]
         == selected["protected_manifest_raw_sha256"] == manifest["protected_manifest_raw_sha256"]
         and digest((project / PUBLIC / "fresh_seed_scan.json").read_bytes()) == plan["fresh_seed_scan_sha256"],
         "Protected old20/30 versions or fresh-seed evidence changed")
    rows = selected["cases"]
    need(len(rows) == len({row["episode_id"] for row in rows}) == selected["actual_cases"], "Every actual attempt must remain exactly once")
    calibration = [row for row in rows if row["stage"] == "calibration"]
    need(len(calibration) == 4 and {row["candidate_id"] for row in calibration} == set(ANGLES)
         and {row["seed"] for row in calibration} == {197304}, "Complete seen-case calibration required")
    options = []
    for index, key in enumerate(ORDER):
        row = next(row for row in calibration if row["candidate_id"] == key)
        if row["valid_evidence"]:
            score = (0, row["physical_action_seconds"]) if row["refine_feasible"] else (1, row["absolute_error_ml"], row["peak_all_label_spill_particles"])
            options.append((score, index, key))
    need(options, "No valid calibration candidate")
    chosen = min(options)[2]
    need(selected["calibration_selected_candidate_id_recomputed"] == chosen
         and next(row for row in calibration if row["candidate_id"] == chosen)["refine_feasible"] is True,
         "Calibration-only time-first candidate selection differs")
    for row in rows:
        stage, key = row["stage"], row["candidate_id"]
        need(stage in stages and row["profile_key"] == key and row["role"] == ("baseline" if key == "baseline" else "candidate")
             and row["profile_namespace"] == "40" and row["retry_enabled"] is True
             and row["profile_id"] == profiles["profile_id"] and row["first40_condition_seed"] == row["seed"]
             and row["condition_seed_provided_by_runner"] is True
             and row["split"] == ("test" if stage == "test" else "development")
             and row["known_seen_development_case"] is (stage == "calibration")
             and row["validation_is_not_final_holdout"] is (stage == "validation")
             and (stage == "calibration" or key == ("baseline" if row["role"] == "baseline" else chosen)),
             "Known/fresh phase, seed, first-angle profile or identical retry mode differs")
        case_check(row, profiles["profiles"][key], {"sha256": digest((project / PUBLIC / "profiles_calibration.json").read_bytes())})
        need(row["second_cycle_execution_or_skip_reason"] == retry3040.retry_reason(row), "Actual retry/skip reason differs")
    for stage in stages:
        group = [row for row in rows if row["stage"] == stage]
        if stage != "calibration":
            seeds = (198101, 198102) if stage == "validation" else range(198301, 198311)
            need(len(group) == COUNTS[stage] and {(r["role"], r["seed"]) for r in group}
                 == {(role, seed) for role in ("baseline", "candidate") for seed in seeds}, "Complete fresh paired grid required")
        for role in ("baseline", "candidate"):
            retry.metric_check(refine, easy, selected["stage_metrics"][stage][role], [r for r in group if r["role"] == role])
    for key in ANGLES:
        retry.metric_check(refine, easy, selected["calibration_metrics_by_candidate"][key], [r for r in calibration if r["candidate_id"] == key])
    lookup = {row["episode_id"]: row for row in rows}
    need(set(selected["paired_comparisons"]) == set(stages) - {"calibration"}, "Actual paired phase coverage differs")
    for stage, pairs in selected["paired_comparisons"].items():
        seeds = (198101, 198102) if stage == "validation" else tuple(range(198301, 198311))
        need(len(pairs) == len(seeds) and {pair["seed"] for pair in pairs} == set(seeds), "Complete actual paired conditions required")
        for pair in pairs:
            for role in ("baseline", "candidate"):
                need(pair[role] == lookup[pair[role]["episode_id"]] and pair[role]["role"] == role
                     and pair[role]["stage"] == stage and pair[role]["seed"] == pair["seed"], "Paired result lost an actual row")
            baseline, candidate = pair["baseline"], pair["candidate"]
            need(abs(pair["candidate_minus_baseline_absolute_error_ml"] - (candidate["absolute_error_ml"] - baseline["absolute_error_ml"])) < 1e-9
                 and abs(pair["candidate_minus_baseline_physical_seconds"] - (candidate["physical_action_seconds"] - baseline["physical_action_seconds"])) < 1e-9
                 and pair["dose_exactly_equal"] is (candidate["received_ml"] == baseline["received_ml"])
                 and pair["physical_duration_exactly_equal"] is (candidate["physical_action_seconds"] == baseline["physical_action_seconds"]),
                 "Actual paired precision/duration comparison differs")
    need(selected["test_pairs"] == selected["paired_comparisons"].get("test", []), "New tests were omitted or invented")
    selection_path = project / PUBLIC / "models/calibration_selection.json"
    selection = read(selection_path)
    need(digest(selection_path.read_bytes()) == selected["calibration_selection"]["sha256"]
         and selection["kind"] == "first40_calibration_selection_v1" and selection["selected_profile_key"] == chosen
         and selection["calibration_gate_passed"] is True and selection["candidate_tie_order"] == list(ORDER)
         and selection["validation_used_for_selection"] is False and selection["test_used_for_selection"] is False
         and selection["feasible_rank_uses_only_physical_time_then_exact_declared_order"] is True,
         "Frozen pre-validation calibration selection changed")
    expected_scores, expected_ranking = {}, []
    for index, key in enumerate(ORDER):
        row = next(row for row in calibration if row["candidate_id"] == key)
        score = (None if not row["valid_evidence"] else [0, row["physical_action_seconds"]] if row["refine_feasible"]
                 else [1, row["absolute_error_ml"], row["peak_all_label_spill_particles"]])
        expected_scores[key] = score
        expected_ranking.append(((2, index) if score is None else (*score, index), key))
    need(selection["candidate_scores"] == expected_scores
         and selection["candidate_ranking"] == [{"profile_key": key, "rank": list(rank)} for rank, key in sorted(expected_ranking)]
         and selection["all_calibration_cases"] == [compact_selection_case(row) for row in calibration]
         and selection["created_at_utc"] == selected["calibration_selection"]["created_at_utc"],
         "Calibration all-case/time-only/exact-tie record changed")
    phase_proof_check(selection["calibration_proof"], "calibration", selected, refine)
    selector = load(project, by_original[HERE + "select_and_freeze.py"]["published_file"], "first40_clone_selector")
    curator = load(project, by_original[HERE + "curate_first_results.py"]["published_file"], "first40_clone_compact_proof")
    expected_matrix = selector.make_matrix("validation", selection, digest(selection_path.read_bytes()))
    need(digest(frozen_json_bytes(expected_matrix, selection_path.read_bytes())) == selected["stage_artifact_sha256"]["validation"]["matrix.json"],
         "Validation matrix no longer binds the frozen calibration choice")
    for model in (selection, *([] if stopped else [read(project / PUBLIC / "models/fixed_strategy.json")])):
        need(model["task_kind"] == "fixed40ml_absolute05ml_24s_onegrain_v1" and model["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
             and model["profiles"] == profiles["profiles"] and model["selected_profile_key"] == chosen
             and model["selected_first_approach_tilt_deg"] == ANGLES[chosen] and model["retry_approach_tilt_deg"] == 87.2
             and model["baseline_profile_key"] == "baseline" and model["baseline_first_approach_tilt_deg"] == 87.1
             and model["both_roles_enable_retry"] is True and model["maximum_cycles"] == 2 and model["only_first_angle_varies"] is True
             and model["fixed_parameters"] == selector.FIXED_PARAMETERS
             and model["source_sha256"] == manifest["source_sha256"] and model["input_sha256"] == manifest["input_sha256"]
             and model["gui_sha256"] == manifest["gui_default_sha256_unchanged"] and model["protected_file_sha256"] == manifest["protected_file_sha256"]
             and model["selector_source_sha256"] == REVIEWED_TOOLS[HERE + "select_and_freeze.py"]
             and model["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_first_run.py"]
             and model["test_used_for_selection"] is False and model["default_gui_deployment"] is False,
             "Frozen checkpoint changed more than its first angle or old protected bindings")
        need(model["calibration_proof"] == selection["calibration_proof"]
             and curator.compact_known_reference(model["known_baseline_equivalence"]) == selected["known_baseline_equivalence"]
             and model["profile_raw_sha256"] == digest((project / PUBLIC / "profiles_calibration.json").read_bytes())
             and model["input_sha256"][model["profile_file"]] == model["profile_raw_sha256"]
             and model["input_sha256"][plan["protected_versions_file"]] == plan["protected_versions_raw_sha256"],
             "Static profile/protected inputs or calibration proof were rebound")
    need(all(refine.timestamp(selection["created_at_utc"]) < refine.timestamp(r["started_at_utc"]) for r in rows if r["stage"] == "validation")
         and refine.timestamp(selection["created_at_utc"]) > refine.timestamp(selection["calibration_proof"]["created_at_utc"]),
         "Calibration selection did not freeze between its audit and validation")
    if stopped:
        receipt = read(project / PUBLIC / "development_stop_receipt.json")
        need(selected["fixed_strategy"] is None and selected["test_pairs"] == []
             and digest((project / PUBLIC / "development_stop_receipt.json").read_bytes()) == selected["finalization"]["sha256"]
             and receipt["kind"] == "first40_finalization_v1" and receipt["outcome"] == "validation_gate_failed"
             and receipt["actual_stages"] == stages and receipt["independent_test_cases"] == 0
             and receipt["no_retune_or_baseline_fallback"] is True, "Stopped development invented a model or tests")
    else:
        path = project / PUBLIC / "models/fixed_strategy.json"
        strategy = read(path)
        phase_proof_check(strategy["validation_proof"], "validation", selected, refine)
        need(digest(path.read_bytes()) == selected["fixed_strategy"]["sha256"]
             and strategy["kind"] == "first40_frozen_first_angle_strategy_v1" and strategy["validation_gate_passed"] is True
             and strategy["calibration_selection_sha256"] == digest(selection_path.read_bytes())
             and strategy["candidate_not_retuned_or_replaced_by_validation"] is True
             and strategy["created_at_utc"] == selected["fixed_strategy"]["created_at_utc"]
             and refine.timestamp(strategy["created_at_utc"]) > refine.timestamp(strategy["validation_proof"]["created_at_utc"])
             and all(refine.timestamp(strategy["created_at_utc"]) < refine.timestamp(r["started_at_utc"]) for r in rows if r["stage"] == "test"),
             "Post-validation strategy did not freeze before fresh tests")
        need(all(r["valid_evidence"] and r["full_home_within_deadline"] and r["peak_all_label_spill_particles"] <= 1
                 and r["no_late_dose_change"] for r in rows if r["stage"] == "validation")
             and all(r["refine_feasible"] for r in rows if r["stage"] == "validation" and r["role"] == "candidate"),
             "Actual fresh validation did not authorize the unchanged candidate")
        expected_matrix = selector.make_matrix("test", selection, digest(selection_path.read_bytes()), digest(path.read_bytes()))
        need(digest(frozen_json_bytes(expected_matrix, path.read_bytes())) == selected["stage_artifact_sha256"]["test"]["matrix.json"],
             "Fresh test matrix retuned or lost its frozen model")
    known_proof_check(selected, manifest, plan)
    prefix_proof_check(selected, manifest, chosen)
    need(selected["old20_and30_versions_unchanged"] is True and selected["bounded_first_angle_calibration"] is True
         and selected["feasible_calibration_rank_uses_only_physical_time_then_declared_order"] is True
         and selected["selected_from_calibration_not_validation_or_test"] is True
         and selected["collector_no_data_fit_or_model_selection_header_scope"] == "collector_only_not_the_finite_calibration_experiment"
         and selected["only_first_angle_varies"] is True and selected["retry_angle_fixed_deg"] == 87.2
         and selected["both_roles_enable_retry"] is True and selected["whole_first_cycle_equality_across_changed_angles_is_not_expected"] is True
         and selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False,
         "Unsupported no-calibration, retry-factor, GUI/default or arbitrary-range claim")
    need(manifest["only_first_angle_varied"] is True and manifest["both_roles_retry_enabled"] is True
         and manifest["fixed_retry_approach_tilt_deg"] == 87.2 and manifest["bounded_calibration_performed"] is True
         and manifest["validation_selected_or_retuned_candidate"] is False and manifest["test_used_for_profile_selection"] is False
         and manifest["original_strict_standard_unchanged"] is True and manifest["default_gui_deployment"] is False
         and manifest["all_continuous_goals_guaranteed"] is False and manifest["raw_logs_or_assets_distributed"] is False,
         "Manifest confused finite first-angle calibration with an unsupported upgrade")
    need(len(manifest["plots"]) == 1, "One actual final figure required")
    for image in manifest["plots"]:
        raw = (project / image["file"]).read_bytes()
        need(image["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == image["sha256"], "Actual selected figure changed")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())

