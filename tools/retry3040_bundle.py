"""Add the completed bounded30/40mL expansion without changing old229 targets."""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

RUN = "droplet_out/pour_policy_retry3040/expansion_001"
HERE = "projects/pour_policy_retry3040/"
PUBLIC = "research/retry3040_20261128/"
REPORT = "results/retry3040_20261128.json"
PLOTS = RUN + "/plots_001"
PLOT_PUBLIC = "results/retry3040_20261128/retry3040_results.png"
KIND = "retry3040_conditional_second_cycle_v1"
GOALS = (30., 40.)
SEEDS = tuple(range(197301, 197311))
COUNTS = {"pilot_001": 4, "pilot_002": 4, "test": 40}
NEW_RUNTIME = tuple(HERE + name for name in ("retry_control.py", "run_retry_test.py", "task_score.py"))
POSTHOC_NAMES = tuple(HERE + name for name in ("freeze_strategy.py", "audit_retry_run.py",
    "audit_retry_experiment.py", "curate_retry_results.py", "plot_retry_results.py", "test_retry_control.py",
    "test_freeze_strategy.py", "test_audit_retry_run.py", "test_curate_retry_results.py"))
REVIEWED_TOOLS = {HERE + name: value for name, value in {
    "freeze_strategy.py": "b7bcd71fa86ff93e24412c390dbafa800f56d58496bc92ab4670239b23c40bd7",
    "audit_retry_run.py": "f658538d89e404eaef4b9ba83d813c64e19d375983dfa0bdcf0fc61c4c1a1d75",
    "audit_retry_experiment.py": "0749edf0fef004192827698b54f7d56397826d41d9b560fe04da2876a872a8bc",
    "curate_retry_results.py": "74e44de93765d0928279d39d17d757088ac9227844f3e903ef0cfc5c0fa2bfa5",
    "plot_retry_results.py": "d0ede62d01c39b782a785d1038fa1c43c4856873b6131cffadd0cd1e4cdadc14",
    "test_retry_control.py": "92616966f05e675e89a5b0158674040f18998d4be2846b9492f8c6631f85eedd",
    "test_freeze_strategy.py": "22d38b78f3fe6b110bfd3f312c93882ef4beacd375082ecfec3f8f7570d14b25",
    "test_audit_retry_run.py": "8b50d8d8aa34e102775ae29d4161e294f4c31493eeded18ee617fe7a59ad1c25",
    "test_curate_retry_results.py": "7f7d33a5036807891b44bf10eb1d068169ba6921351a3989b4320c9fc2c18c59",
}.items()}
ATTRIBUTES = ("# Preserve original frozen source, profile and strategy SHA.\n"
    "* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n"
    ".gitattributes text eol=lf\nREADME.md text eol=lf\nsource_manifest.json text eol=lf\n"
    "*.py -text\n*.json -text\nsource_manifest.json text eol=lf\n")
README = ("# Bounded 30 and 40 mL feedback expansion\n\n"
    "This study targets 30 and 40 mL within +/-0.5 mL in 24 raw physical seconds, including return to "
    "the original pregrasp position and 12 stable frames. Peak spill across all particle labels must "
    "remain at most 1 particle; another 2 seconds checks late water. The +/-1 mL secondary criterion "
    "and original +/-0.1% criteria (0.03/0.04 mL) remain separate. See "
    "[the actual selected results](../../results/retry3040_20261128.json).\n\n"
    "Both modes use the same target-specific first approach. Baseline permits one correction cycle; "
    "candidate additionally permits one conditional retry after an actual settled undershoot, current "
    "spill check and conservative full-home time check. All motion speeds remain fixed. Retry intent, "
    "actual cycle starts, approach drainage and skipped micro actions are recorded separately. A time "
    "budget rejection is not an executed second cycle, and an overdose cannot be removed by adding water.\n\n"
    "All eight development cases are retained, including the failed first 40 mL proposal. Only the "
    "preregistered signed 0.1 degree adjustment is allowed: retain a passing goal, lower an overdosing "
    "goal or raise an underdosing goal. Both angles move together. After the second four-case stage "
    "passes its gate, one fixed strategy is frozen before forty fresh tests: ten matched initial-state "
    "pairs for each goal. Test outcomes never select or retune the profile. This is a bounded feedback "
    "engineering experiment, not a new PPO or neural-policy training result.\n\n"
    "Three runtime files reference 25 unchanged V12 dependencies. The old 20 mL controllers and results "
    "are preserved and are not imported by this runtime. Initial and final profile JSON files, both "
    "development matrices, the signed adjustment receipt and frozen strategy are included. Profiles "
    "are small self-written inputs; the six geometric cache/asset inputs remain local. Restore the "
    "original module layout and licensed resources for a new protocol; do not rebind SHA guards or "
    "resume the author's private journals.\n\n"
    "Audit dependencies are referenced from the prior V12/V13 archives: "
    "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py, audit_response_collection.py "
    "and cycle_samples.py, plus projects/pour_policy_training_v9/audit_fine_run.py from the earlier V9 "
    "reference. Their original relative paths and literal SHA guards are required. The documented "
    "inherited diagnostic substep counter at time zero is separate from checked completed solver frames.\n\n"
    "One posthoc test uses an explicit synthetic-private-path placeholder in its public failure fixture "
    "and removal assertion. It still tests error-text sanitization; the manifest records separate "
    "original and published SHA values. "
    "All runtime, strategy, profile and result bytes remain unchanged.\n\n"
    "No SDK, third-party robot assets, NPZ, raw step logs, videos or credentials are distributed. "
    "Kinematic pregrasp and particle mass truth are not contact-grasp, real-scale or hardware validation. "
    "Two fixed targets and narrow initial-state randomization do not guarantee arbitrary 20--40 mL. "
    "No GUI default or out-of-box GUI entry is replaced.\n")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_path_anchors():
    # Keep the negative sanitizer test, with an explicit public placeholder rather than a fake path.
    return (
        (b'"Synthetic failure in ' + b'C' + b':/private/person/source.py"',
         b'"Synthetic failure in <synthetic-private-path>/person/source.py"'),
        (b'self.assertNotIn("' + b'C' + b':/private", json.dumps(case))',
         b'self.assertNotIn("<synthetic-private-path>", json.dumps(case))'),
    )


def public_posthoc_bytes(original, raw):
    if original != HERE + "test_curate_retry_results.py":
        return raw, "unchanged_raw_bytes"
    for old, new in test_path_anchors():
        need(raw.count(old) == 1 and new not in raw, "Synthetic-path test conversion anchor changed")
        raw = raw.replace(old, new)
    return raw, "synthetic_private_path_test_fixture_placeholder_only"


def original_posthoc_bytes(original, raw, transform):
    if transform == "unchanged_raw_bytes":
        need(original != HERE + "test_curate_retry_results.py", "Declared test conversion was omitted")
        return raw
    need(original == HERE + "test_curate_retry_results.py"
         and transform == "synthetic_private_path_test_fixture_placeholder_only", "Unreviewed publication transform")
    for old, new in test_path_anchors():
        need(raw.count(new) == 1 and old not in raw, "Published synthetic-path test anchor changed")
        raw = raw.replace(new, old)
    return raw


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
    # Do not inspect held-out journals or frozen models until all48 are complete.
    for stage, count in COUNTS.items():
        summary, job = read(workspace / RUN / stage / "summary.json"), read(workspace / RUN / stage / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and job.get("state") == "complete" and job.get("pid") is None, "Complete4/4/40 study required")
    need(set(REVIEWED_TOOLS) == set(POSTHOC_NAMES), "3040 posthoc allowlist differs")
    for name, expected in REVIEWED_TOOLS.items():
        need(isinstance(expected, str) and len(expected) == 64 and digest((workspace / name).read_bytes()) == expected,
             "Final reviewed3040 tool required: " + name)
    base = load(project, "tools/adaptive_response_bundle.py", "retry3040_parent_publication")
    curator = load(workspace, HERE + "curate_retry_results.py", "retry3040_reviewed_curator")
    audit_path = workspace / RUN / "overall_audit.json"
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("kind") == KIND
         and audit["development_stop"] is False and audit["outcome"] == "fresh_test_completed"
         and audit["actual_development_stages"] == ["pilot_001", "pilot_002"]
         and audit["auditor_source_file"] == HERE + "audit_retry_experiment.py"
         and audit["auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_retry_experiment.py"], "Full independent48-case audit required")
    selected = read(project / REPORT)
    need(same_summary(selected, curator.build_summary(workspace / RUN, audit_path, digest(audit_path.read_bytes()))),
         "Selected3040 report differs from all actual audited attempts")
    base.private_paths((project / REPORT).read_bytes())
    return base, selected


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base, selected = gate(workspace, project)
    prior = dict(prior_prepared)
    parent_bytes = prior[base.PUBLIC + "source_manifest.json"]
    entries = {e["original_file"]: e for e in json.loads(parent_bytes)["files"] if e["role"] == "collection_runtime"}
    shared = set(entries) - {"projects/pour_policy_adaptive_learning/collect_cycles.py"}
    protocols = {stage: read(workspace / RUN / stage / "protocol.json") for stage in COUNTS}
    final = protocols["pilot_002"]
    sources = final["source_sha256"]
    need(len(shared) == 25 and len(sources) == 28 and set(sources) == shared | set(NEW_RUNTIME), "Exact25 shared/3 new source inventory required")
    bindings = []
    for name in sorted(shared):
        entry = entries[name]
        need(sources[name] == entry["original_sha256"] and digest(prior[entry["published_file"]]) == entry["published_sha256"], "Shared V12 source changed")
        bindings.append(entry)
    core = lambda p: {k: v for k, v in p["input_sha256"].items() if k != p["matrix"]["goal_profile_file"]}
    for protocol in protocols.values():
        need(protocol["source_sha256"] == sources and protocol["gui_sha256"] == final["gui_sha256"]
             and len(protocol["input_sha256"]) == 7 and core(protocol) == core(final), "Sources/geometry/default GUI changed")
    need(protocols["test"]["input_sha256"] == final["input_sha256"], "Fresh tests changed final frozen profile input")
    files, mapping = [], []

    def add(original, path, role, expected, destination=None):
        raw = path.read_bytes()
        need(digest(raw) == expected, "Explicit3040 publication file changed: " + original)
        transform = "unchanged_raw_bytes"
        if role == "posthoc_freeze_audit_or_test":
            raw, transform = public_posthoc_bytes(original, raw)
        base.private_paths(raw)
        destination = destination or PUBLIC + original
        if destination.endswith(".py"): ast.parse(raw.decode("utf8"), filename=destination)
        files.append((destination, raw))
        mapping.append({"role": role, "original_file": original, "published_file": destination,
            "original_sha256": expected, "published_sha256": digest(raw), "transform": transform})

    for name in NEW_RUNTIME:
        add(name, workspace / RUN / "pilot_001/source_snapshot" / name, "runtime", sources[name])
    for name, expected in REVIEWED_TOOLS.items(): add(name, workspace / name, "posthoc_freeze_audit_or_test", expected)
    metadata = [("experiment_plan.json", workspace / RUN / "experiment_plan.json")]
    for stage in ("pilot_001", "pilot_002"):
        metadata.append((stage + "_matrix.json", workspace / RUN / stage / "matrix.json"))
        name = protocols[stage]["matrix"]["goal_profile_file"]
        metadata.append(("profiles_" + stage + ".json", workspace / RUN / stage / "input_snapshot" / name))
    receipt = selected["development_adjustment"]
    metadata.append(("development_adjustment_001.json", base.inside(workspace / RUN, receipt["file"])))
    for name, path in metadata: add(name, path, "plan_profile_or_adjustment", digest(path.read_bytes()), PUBLIC + name)
    fixed = selected["fixed_strategy"]
    add(fixed["file"], base.inside(workspace / RUN, fixed["file"]), "fixed_research_strategy", fixed["sha256"], PUBLIC + "models/fixed_strategy.json")
    plot = read(workspace / PLOTS / "plot_provenance.json")
    plot_report = workspace / PLOTS / "retry3040_results.json"
    image = (workspace / PLOTS / "retry3040_results.png").read_bytes()
    need(plot["complete"] is True and plot["synthetic_fixture"] is False
         and plot["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and plot["curator_source_sha256"] == REVIEWED_TOOLS[HERE + "curate_retry_results.py"]
         and plot["plot_source_sha256"] == REVIEWED_TOOLS[HERE + "plot_retry_results.py"]
         and same_summary(read(plot_report), selected) and digest(plot_report.read_bytes()) == plot["files_sha256"]["retry3040_results.json"]
         and image.startswith(b"\x89PNG\r\n\x1a\n") and digest(image) == plot["files_sha256"]["retry3040_results.png"], "Real final3040 plot/input provenance differs")
    files.append((PLOT_PUBLIC, image))
    manifest = {"schema_version": 1, "kind": "retry3040_bounded_expansion_publication_v1",
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"], "experiment_plan_sha256": selected["experiment_plan_sha256"],
        "budget": selected["budget"], "source_count": 28, "shared_source_count": 25, "source_sha256": sources,
        "stage_input_sha256": {stage: p["input_sha256"] for stage, p in protocols.items()},
        "stage_artifact_sha256": selected["stage_artifact_sha256"], "files": mapping, "shared_source_bindings": bindings,
        "parent_adaptive_manifest_sha256": digest(parent_bytes), "parent_manifests_unchanged": True,
        "plots": [{"file": PLOT_PUBLIC, "sha256": digest(image)}], "gui_default_sha256_unchanged": final["gui_sha256"],
        "original_strict_standard_unchanged": True, "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "test_used_for_profile_selection": False, "no_test_based_model_or_profile_selection": True,
        "bounded_development_profile_calibration": True, "raw_logs_or_assets_distributed": False}
    raw = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(raw)
    files.extend(((PUBLIC + "source_manifest.json", raw), (PUBLIC + "README.md", README.encode()), (PUBLIC + ".gitattributes", ATTRIBUTES.encode())))
    need(len({name for name, _ in files}) == len(files) == 23 and not set(name for name, _ in files) & set(prior), "Only23 new3040 targets may be added")
    return files


def case_check(row, goal_profiles, profile_input):
    goal, amount, elapsed = row["target_ml"], row["received_ml"], row["physical_action_seconds"]
    flags = ("valid_evidence", "mass_conserved", "all_states_finite", "full_home_within_deadline",
        "no_late_dose_change", "refine_feasible", "easy_feasible", "original_strict_feasible",
        "original_strict_dose_passed", "retry_enabled", "actual_second_cycle_started", "retry_allowed")
    need(all(type(row[key]) is bool for key in flags)
         and type(row["peak_all_label_spill_particles"]) is int and row["peak_all_label_spill_particles"] >= 0,
         "Task/physics flags and actual all-label spill count must be explicit")
    need(goal in GOALS and all(type(v) in (int, float) and math.isfinite(v) for v in (amount, elapsed, row["error_ml"], row["absolute_error_ml"]))
         and abs(row["error_ml"] - (amount - goal)) < 1e-9 and abs(row["absolute_error_ml"] - abs(amount - goal)) < 1e-9, "Actual3040 volume/error changed")
    safe = bool(row["valid_evidence"] and row["mass_conserved"] and row["all_states_finite"] and row["full_home_within_deadline"]
        and elapsed <= 24. and row["peak_all_label_spill_particles"] <= 1 and row["no_late_dose_change"])
    need(row["refine_feasible"] is bool(safe and goal - .5 <= amount <= goal + .5)
         and row["easy_feasible"] is bool(safe and goal - 1. <= amount <= goal + 1.)
         and row["original_strict_dose_passed"] is (goal - goal * .001 <= amount <= goal + goal * .001)
         and row["original_strict_feasible"] is bool(safe and goal - goal * .001 <= amount <= goal + goal * .001), "Three target-specific closed gates differ")
    expected = goal_profiles[format(goal, "g")]
    need(row["retry_enabled"] is (row["role"] == "candidate") and row["profile_key"] == format(goal, "g")
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
             and d["target_profile_id"] == row["profile_id"] and d["declared_retry_approach_tilt_deg"] == expected["retry_approach_tilt_deg"]
             and d["completed_cycles_before_decision"] == 1 and d["retry_allowed"] is allowed and d["spill_guard_passed"] is spill
             and d["target_profile_key"] == row["profile_key"] and d["deficit_exceeds_half_ml"] is (d["actual_received_ml"] < goal - .5)
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


def pair_proof_check(selected, manifest):
    """Bind every available first-cycle or no-cycle anchor proof to real cases."""
    rows = selected["cases"]
    proofs = selected["paired_first_cycle_equivalence"]
    expected = {(stage, goal, seed) for stage in COUNTS for goal in GOALS
                for seed in (SEEDS if stage == "test" else (197101,))}
    need(len(proofs) == 24 and {(p["stage"], p["target_ml"], p["seed"]) for p in proofs} == expected,
         "All24 actual development/test first-cycle pair comparisons are required")
    for proof in proofs:
        group = {r["role"]: r for r in rows if (r["stage"], r["target_ml"], r["seed"])
                 == (proof["stage"], proof["target_ml"], proof["seed"])}
        need(set(group) == {"baseline", "candidate"} and type(proof["verified"]) is bool
             and type(proof["endpoint_unavailable"]) is bool, "Pair proof identity/availability differs")
        if proof["endpoint_unavailable"]:
            need(proof["verified"] is False and proof["all_completed_frame_substep_counts_exactly_equal"] is None
                 and any(not r["actual_cycles"] or not r["actual_cycles"][0]["has_recorded_endpoint"] for r in group.values()),
                 "Unavailable paired endpoints cannot be relabelled as equal physics")
            continue
        kind = proof["comparison_endpoint_kind"]
        need(kind in ("actual_first_complete_or_unsettled_whole_cycle", "actual_quiet_coarse_anchor_no_cycle_needed")
             and type(proof["original_first_cycle_events_exactly_equal"]) is bool
             and type(proof["actual_water_motion_partitions_exactly_equal"]) is bool
             and proof["verified"] is bool(proof["original_first_cycle_events_exactly_equal"]
                 and proof["actual_water_motion_partitions_exactly_equal"])
             and proof["all_completed_frame_substep_counts_exactly_equal"] is
                 (True if proof["actual_water_motion_partitions_exactly_equal"] else None)
             and proof["completed_frame_substep_equality_basis"] == "included_in_independent_physical_equality_conjunction",
             "Audited event/actual-physics/substep conjunction differs")
        for role, row in group.items():
            need(proof[role + "_episode_id"] == row["episode_id"], "Pair proof lost its actual case identity")
            if kind == "actual_quiet_coarse_anchor_no_cycle_needed":
                need(not row["actual_cycles"] and row["first_quiet_anchor_available"] is True,
                     "A no-cycle anchor cannot replace an actually executed cycle")
                amount, elapsed = row["first_quiet_anchor_received_ml"], row["first_quiet_anchor_physical_seconds"]
            else:
                need(row["actual_cycles"] and row["actual_cycles"][0]["has_recorded_endpoint"] is True,
                     "Actual paired first endpoint was lost")
                cycle = row["actual_cycles"][0]
                amount, elapsed = cycle["actual_end_received_ml"], cycle["end_physical_s"]
                need(abs(proof[role + "_actual_first_cycle_end_received_ml"] - amount) < 1e-9,
                     "First-cycle outcome was replaced by a later dose")
            need(abs(proof[role + "_actual_comparison_received_ml"] - amount) < 1e-9,
                 "Actual pair comparison dose differs")
            if proof["verified"]:
                need(abs(proof["first_cycle_end_physical_s"] - elapsed) < 1e-9,
                     "Verified paired endpoint physical time differs")
        if proof["verified"]:
            need(proof["baseline_actual_comparison_received_ml"] == proof["candidate_actual_comparison_received_ml"],
                 "Verified first-cycle physical doses differ")
        need(type(proof["first_cycle_frames"]) is int and proof["first_cycle_frames"] > 0,
             "Actual first-cycle frame count is required")
        counter = proof["zero_time_inherited_last_substeps"]
        need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True
             and counter["source_file"] == "assets/robot_testtube_pour/scripts/liquid.py"
             and counter["source_sha256"] == manifest["source_sha256"][counter["source_file"]]
             and type(counter["baseline"]) is int and type(counter["candidate"]) is int
             and type(counter["reason"]) is str and bool(counter["reason"]),
             "Inherited t0 diagnostic counter cannot hide a completed solver update")


def retry_reason(row):
    """Keep actual second-cycle execution distinct from skip and guard outcomes."""
    cycles, decisions, goal = row["actual_cycles"], row["retry_decisions"], row["target_ml"]
    if row["actual_second_cycle_started"]: return "second_cycle_actually_started"
    if not row["retry_enabled"]: return "baseline_declares_one_cycle_only"
    if row["retry_allowed"]: return "retry_allowed_intent_but_second_start_not_recorded"
    if not cycles:
        if row["first_quiet_anchor_settled_measurement"] is not True: return "actual_coarse_anchor_unavailable_or_not_settled"
        amount = row["first_quiet_anchor_received_ml"]
        if goal - .5 <= amount <= goal + .5: return "actual_coarse_settled_band"
        if amount > goal + .5: return "actual_coarse_settled_overdose_cannot_be_removed_by_retry"
        return "actual_coarse_settled_undershoot_without_first_cycle"
    first = cycles[0]
    if not first["has_recorded_endpoint"]: return "actual_first_cycle_endpoint_unavailable"
    if first["settled"] is not True: return "actual_first_cycle_endpoint_unsettled"
    amount = first["actual_end_received_ml"]
    if amount > goal + .5: return "actual_first_cycle_overdose_cannot_be_removed_by_retry"
    if goal - .5 <= amount <= goal + .5: return "actual_first_cycle_already_in_primary_band"
    if decisions and not decisions[0]["actual_settled_measurement"]: return "actual_remeasurement_not_quiet"
    if decisions and not decisions[0]["spill_guard_passed"]: return "actual_current_spill_guard_failed"
    if decisions and not decisions[0]["full_home_budget_fits"]: return "insufficient_conservative_full_home_time_budget"
    return "measured_undershoot_without_recorded_retry_permission"


def verify_public(project):
    """Clone-safe checks of all48 attempts; no private logs, SDK or test selection."""
    project = Path(project).resolve()
    base = load(project, "tools/adaptive_response_bundle.py", "retry3040_clone_parent")
    retry = load(project, "tools/retry20_bundle.py", "retry3040_clone_metrics")
    refine = load(project, "tools/refine20_bundle.py", "retry3040_clone_reference")
    easy = load(project, "tools/easy20_bundle.py", "retry3040_clone_easy")
    selected, manifest = read(project / REPORT), read(project / PUBLIC / "source_manifest.json")
    need(selected["complete"] is True and selected["kind"] == KIND and selected["outcome"] == "fresh_test_completed"
         and selected["development_stop"] is False and selected["independent_test_not_run"] is False
         and selected["actual_stages"] == list(COUNTS) and selected["actual_development_stages"] == ["pilot_001", "pilot_002"]
         and selected["budget"] == {"development": 8, "test": 40, "total": 48, "maximum_development": 8, "maximum_total": 48}, "Complete bounded48-case report required")
    need(manifest["kind"] == "retry3040_bounded_expansion_publication_v1" and manifest["report_file"] == REPORT
         and manifest["report_sha256"] == digest((project / REPORT).read_bytes()) and manifest["source_count"] == 28 and manifest["shared_source_count"] == 25
         and manifest["budget"] == selected["budget"] and manifest["parent_manifests_unchanged"] is True
         and manifest["independent_audit_raw_sha256"] == selected["independent_audit_raw_sha256"]
         and manifest["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
         and manifest["stage_artifact_sha256"] == selected["stage_artifact_sha256"]
         and manifest["parent_adaptive_manifest_sha256"] == digest((project / base.PUBLIC / "source_manifest.json").read_bytes()), "New/old report-source provenance differs")
    mapping = manifest["files"]
    expected = {PUBLIC + name for name in ("source_manifest.json", "README.md", ".gitattributes")}
    names = lambda role: {entry["original_file"] for entry in mapping if entry["role"] == role}
    metadata_names = {"experiment_plan.json", "pilot_001_matrix.json", "pilot_002_matrix.json", "profiles_pilot_001.json", "profiles_pilot_002.json", "development_adjustment_001.json"}
    need(len(mapping) == 19 and names("runtime") == set(NEW_RUNTIME) and names("posthoc_freeze_audit_or_test") == set(POSTHOC_NAMES)
         and names("plan_profile_or_adjustment") == metadata_names and names("fixed_research_strategy") == {"fixed_strategy.json"}, "Explicit3040 allowlist differs")
    for entry in mapping:
        name = entry["published_file"]
        need(name.startswith(PUBLIC) and (project / name).resolve().is_relative_to(project / PUBLIC), "Archive target escaped")
        raw = (project / name).read_bytes()
        original = (original_posthoc_bytes(entry["original_file"], raw, entry["transform"])
                    if entry["role"] == "posthoc_freeze_audit_or_test" else raw)
        need(digest(raw) == entry["published_sha256"] and digest(original) == entry["original_sha256"],
             "Frozen original or published3040 bytes changed")
        if entry["role"] != "posthoc_freeze_audit_or_test":
            need(entry["transform"] == "unchanged_raw_bytes", "Runtime/model/metadata must retain original bytes")
        if entry["role"] == "runtime": need(manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Runtime binding differs")
        if entry["role"] == "posthoc_freeze_audit_or_test": need(REVIEWED_TOOLS[entry["original_file"]] == entry["original_sha256"], "Reviewed posthoc binding differs")
        base.private_paths(raw)
        if name.endswith(".py"): compile(raw.decode("utf8"), name, "exec")
        expected.add(name)
    actual = {p.relative_to(project).as_posix() for p in (project / PUBLIC).rglob("*") if p.is_file() and not (p.parent.name == "__pycache__" and p.suffix == ".pyc")}
    need(actual == expected and len(expected) == 22, "Extra SDK/cache/log or missing3040 archive file")
    bindings = manifest["shared_source_bindings"]
    need(len(bindings) == len({e["original_file"] for e in bindings}) == 25 and len(manifest["source_sha256"]) == 28
         and set(manifest["source_sha256"]) == set(NEW_RUNTIME) | {e["original_file"] for e in bindings}
         and not any("/pour_policy_retry20/" in e["original_file"] or "/pour_policy_refine20/" in e["original_file"] for e in bindings), "Old20 controller became a runtime ancestor")
    for entry in bindings:
        need(digest((project / entry["published_file"]).read_bytes()) == entry["published_sha256"]
             and manifest["source_sha256"][entry["original_file"]] == entry["original_sha256"], "Shared V12 dependency changed")
    need(len(manifest["gui_default_sha256_unchanged"]) == 3, "Three unchanged old GUI weights are required")
    for name, expected_sha in manifest["gui_default_sha256_unchanged"].items():
        need(digest((project / name).read_bytes()) == expected_sha, "An old GUI default was replaced")
    plan_path = project / PUBLIC / "experiment_plan.json"
    plan = read(plan_path)
    need(digest(plan_path.read_bytes()) == selected["experiment_plan_sha256"] and plan["draft"] is False
         and plan["targets_ml"] == list(GOALS) and plan["test"]["seeds"] == list(SEEDS)
         and plan["development"]["maximum_cases"] == 8 and plan["test"]["cases"] == 40,
         "Bounded preregistration changed")
    need(selected["task_contract"] == {"targets_ml": list(GOALS), **plan["task"], "secondary_easy_absolute_tolerance_ml": 1.,
        "original_strict_absolute_tolerance_ml": {"30": .03, "40": .04}}, "Primary or target-specific strict criterion changed")
    need(selected["physics"] == {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
        "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
        "total_particles": 11625, "source_particles": 2880,
        "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "Actual solver/particle units or measurement limitations changed")
    inputs = selected["stage_goal_profile_inputs"]
    profiles = selected["stage_goal_profiles"]
    profile_ids = {}
    for stage in ("pilot_001", "pilot_002"):
        raw_profile = project / PUBLIC / ("profiles_" + stage + ".json")
        profile = read(raw_profile)
        profile_ids[stage] = profile["profile_id"]
        need(digest(raw_profile.read_bytes()) == inputs[stage]["sha256"]
             and profile["goal_profiles"] == profiles[stage] and profile["all_motion_speeds_fixed"] is True
             and digest((project / PUBLIC / (stage + "_matrix.json")).read_bytes()) == selected["stage_artifact_sha256"][stage]["matrix.json"], "Development matrix/profile changed")
    need(inputs["test"] == inputs["pilot_002"] and profiles["test"] == profiles["pilot_002"], "Fresh tests retuned the profile")
    need(inputs["pilot_001"]["file"] == plan["initial_goal_profile_file"]
         and inputs["pilot_001"]["sha256"] == plan["initial_goal_profile_raw_sha256"], "Initial engineering profile differs from preregistration")
    profile_ids["test"] = profile_ids["pilot_002"]
    rows = selected["cases"]
    need(len(rows) == len({row["episode_id"] for row in rows}) == 48, "Every attempted development/test case must remain")
    for stage, count in COUNTS.items():
        group = [row for row in rows if row["stage"] == stage]
        seeds = SEEDS if stage == "test" else (197101,)
        need(len(group) == count and {(r["target_ml"], r["role"], r["seed"]) for r in group}
             == {(g, role, seed) for g in GOALS for role in ("baseline", "candidate") for seed in seeds}, "Complete goal/role/seed grid required")
        for role in ("baseline", "candidate"):
            retry.metric_check(refine, easy, selected["stage_metrics"][stage][role], [r for r in group if r["role"] == role])
            for goal in GOALS:
                retry.metric_check(refine, easy, selected["per_goal_metrics"][format(goal, "g")][stage][role],
                    [r for r in group if r["role"] == role and r["target_ml"] == goal])
    for row in rows:
        need(row["split"] == ("test" if row["stage"] == "test" else "development")
             and row["development_not_independent_test"] is (row["stage"] != "test")
             and row["profile_id"] == profile_ids[row["stage"]], "Development/profile identity relabelled")
        case_check(row, profiles[row["stage"]], inputs[row["stage"]])
        need(row["second_cycle_execution_or_skip_reason"] == retry_reason(row), "Second-cycle outcome/guard reason differs")
    freeze_entry = next(e for e in mapping if e["original_file"] == HERE + "freeze_strategy.py")
    freezer = load(project, freeze_entry["published_file"], "retry3040_clone_freeze_rules")
    old_results = freezer.candidate_results([{"role": r["role"], "target_ml": r["target_ml"], "result": {"received_ml": r["received_ml"]}}
        for r in rows if r["stage"] == "pilot_001"])
    freezer.validate_adjustment(profiles["pilot_001"], profiles["pilot_002"], old_results)
    need(all(r["valid_evidence"] and r["full_home_within_deadline"] and r["peak_all_label_spill_particles"] <= 1 and r["no_late_dose_change"]
             for r in rows if r["stage"] != "test")
         and all(r["refine_feasible"] for r in rows if r["stage"] == "pilot_002" and r["role"] == "candidate"), "Actual development gate failed")
    strategy_path = project / PUBLIC / "models/fixed_strategy.json"
    strategy = read(strategy_path)
    need(digest(strategy_path.read_bytes()) == selected["fixed_strategy"]["sha256"]
         and strategy["kind"] == "fixed3040_conditional_second_cycle_strategy_v1"
         and strategy["completed_development_stages"] == ["pilot_001", "pilot_002"]
         and strategy["development_gate_passed"] is True and strategy["goal_profiles"] == profiles["test"]
         and strategy["goal_profile_raw_sha256"] == inputs["test"]["sha256"]
         and strategy["source_sha256"] == manifest["source_sha256"]
         and strategy["input_sha256"] == manifest["stage_input_sha256"]["test"]
         and strategy["gui_sha256"] == manifest["gui_default_sha256_unchanged"]
         and strategy["no_new_training_search_or_neural_policy"] is True and strategy["test_used_for_selection"] is False
         and strategy["default_gui_deployment"] is False and strategy["freeze_helper_sha256"] == REVIEWED_TOOLS[HERE + "freeze_strategy.py"], "Frozen fixed strategy changed")
    need(strategy["fixed_parameters"] == {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
        "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6., "approach_tilt_deg_s": 6.,
        "action_delta_deg": .03, "action_tilt_deg_s": .6, "hold_seconds": .1, "backoff_delta_deg": .5,
        "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6., "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}
         and strategy["maximum_cycles_baseline"] == 1 and strategy["maximum_cycles_candidate"] == 2
         and strategy["test_cases"] == 40 and strategy["test_goal_seed_conditions"] == 20
         and strategy["experiment_plan_sha256"] == selected["experiment_plan_sha256"]
         and strategy["stage_auditor_source_sha256"] == REVIEWED_TOOLS[HERE + "audit_retry_run.py"],
         "Frozen motion speeds, physical measurement timing or case budget changed")
    need(digest((project / PUBLIC / "development_adjustment_001.json").read_bytes()) == selected["development_adjustment"]["sha256"]
         == strategy["development_adjustment"]["sha256"], "Signed adjustment receipt changed")
    need([record["stage"] for record in strategy["development_stages"]] == ["pilot_001", "pilot_002"]
         and strategy["development_cases"] == 8 and strategy["selected_development_stage"] == "pilot_002",
         "Both actual development stages must remain in the frozen strategy")
    freezer.validate_adjustment_receipt(read(project / PUBLIC / "development_adjustment_001.json"),
        strategy["development_stages"][0], strategy["development_stages"][1])
    for record in strategy["development_stages"]:
        stage = record["stage"]
        need(record["audit_sha256"] == selected["stage_artifact_sha256"][stage]["audit.json"]
             and {**record["input_artifact_sha256"], "audit.json": record["audit_sha256"]} == selected["stage_artifact_sha256"][stage], "Strategy discarded previous development evidence")
        group = [r for r in rows if r["stage"] == stage]
        expected_results = freezer.candidate_results([{"role": r["role"], "target_ml": r["target_ml"],
            "result": {"received_ml": r["received_ml"]}} for r in group])
        need(record["candidate_by_goal"] == expected_results and record["goal_profiles"] == profiles[stage]
             and record["goal_profile_raw_sha256"] == inputs[stage]["sha256"]
             and record["goal_profile_id"] == profile_ids[stage]
             and record["all_non_dose_safety_passed"] is True
             and record["development_gate_passed"] is all(r["primary_passed"] for r in expected_results.values()),
             "Frozen actual development doses/profile gates differ")
    need(refine.timestamp(strategy["created_at_utc"]) > max(refine.timestamp(r["completed_at_utc"]) for r in rows if r["stage"] != "test")
         and all(refine.timestamp(strategy["created_at_utc"]) < refine.timestamp(r["started_at_utc"]) for r in rows if r["stage"] == "test"), "Strategy did not freeze between development and fresh tests")
    lookup = {r["episode_id"]: r for r in rows}
    pairs = selected["test_pairs"]
    need(len(pairs) == 20 and {(p["target_ml"], p["seed"]) for p in pairs} == {(g, seed) for g in GOALS for seed in SEEDS}, "Twenty independent paired conditions required")
    for p in pairs:
        for role in ("baseline", "candidate"):
            need(p[role] == lookup[p[role]["episode_id"]] and p[role]["role"] == role and p[role]["stage"] == "test"
                 and p[role]["target_ml"] == p["target_ml"] and p[role]["seed"] == p["seed"], "Paired actual result changed")
        need(p["final_dose_exactly_equal"] is (p["baseline"]["received_ml"] == p["candidate"]["received_ml"])
             and p["physical_duration_exactly_equal"] is
                 (p["baseline"]["physical_action_seconds"] == p["candidate"]["physical_action_seconds"]),
             "Actual paired final dose/time equality claim differs")
    for goal in GOALS:
        group = [p for p in pairs if p["target_ml"] == goal]
        no_retry = [p for p in group if not p["candidate"]["actual_second_cycle_started"]]
        need(selected["paired_no_retry_matching"][format(goal, "g")] == {
            "fresh_pairs": 10, "no_retry_pairs": len(no_retry),
            "no_retry_dose_and_time_exact_pairs": sum(p["final_dose_exactly_equal"] and p["physical_duration_exactly_equal"] for p in no_retry),
            "actual_second_cycle_seeds": [p["seed"] for p in group if p["candidate"]["actual_second_cycle_started"]],
            "baseline_failure_candidate_pass_pairs": sum(not p["baseline"]["refine_feasible"] and p["candidate"]["refine_feasible"] for p in group),
            "baseline_pass_candidate_failure_pairs": sum(p["baseline"]["refine_feasible"] and not p["candidate"]["refine_feasible"] for p in group)},
            "Target-specific retry branch coverage or unchanged-pair claim differs")
    pair_proof_check(selected, manifest)
    need(selected["test_used_for_profile_selection"] is False and selected["development_is_not_independent_test"] is True
         and selected["all_motion_speeds_fixed"] is True and selected["old20_not_imported_by_new_runtime"] is True
         and selected["bounded_development_angle_calibration_performed"] is True
         and selected["all_physical_sources_geometry_inputs_gui_unchanged"] is True
         and selected["only_declared_goal_profile_inputs_may_differ"] is True
         and selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and manifest["test_used_for_profile_selection"] is False and manifest["no_test_based_model_or_profile_selection"] is True
         and manifest["bounded_development_profile_calibration"] is True and manifest["raw_logs_or_assets_distributed"] is False,
         "Unsupported calibration/test-selection/default/range claim")
    need(len(manifest["plots"]) == 1, "One real selected figure required")
    for image in manifest["plots"]:
        raw = (project / image["file"]).read_bytes()
        need(image["file"] == PLOT_PUBLIC and raw.startswith(b"\x89PNG\r\n\x1a\n") and digest(raw) == image["sha256"], "Published real figure changed")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / PUBLIC / "source_manifest.json").read_bytes())
