"""Final independent evidence/selection audit for coarse-dose-speed-only40mL study.

Three seen cases calibrate one first approach angle at fixed coarse speed. Fresh validation gates a previously
frozen choice without retuning or baseline fallback. Fresh tests never select.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_balanced40/"
STAGE_SHA = "d25ada622aefab3e39afd7ba6c6f986edcd7a0e53c1b6015307b57f988dd883e"
MAPS = ("source_sha256", "input_sha256", "gui_sha256", "protected_file_sha256")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def load_stage(root=ROOT):
    path = root / HERE / "audit_balanced_run.py"
    need(sha(path) == STAGE_SHA, "Frozen independently executed stage auditor changed")
    spec = importlib.util.spec_from_file_location("balanced40_pinned_stage_auditor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def actual_stages(experiment, development_stop):
    if development_stop:
        need(not (experiment / "test").exists() and not (experiment / "fixed_strategy.json").exists(),
             "A stopped development study cannot hide tests or a post-validation strategy")
        names = ["calibration", *(["validation"] if (experiment / "validation").exists() else [])]
    else:
        names = ["calibration", "validation", "test"]
    return names


def complete_gate(experiment, names, stage):
    """No journal/checkpoint/test read before every actual required phase is complete."""
    for name in names:
        run = experiment / name
        need((run / "summary.json").is_file() and (run / "job_state.json").is_file(),
             "Every actual required stage must complete before final audit")
        summary, job = read(run / "summary.json"), read(run / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == stage.COUNTS[name]
             and summary.get("stage") == name and job.get("state") == "complete" and job.get("pid") is None
             and not (run / "worker.lock").exists(), "Every actual required stage must complete before final audit")


def verify_cached(experiment, name, report):
    cached = read(experiment / name / "audit.json")
    need(cached["complete"] is True and cached["pass"] is True and cached["stage"] == name
         and cached["auditor_source_sha256"] == STAGE_SHA, "Cached independent stage identity differs")
    for key, value in report.items():
        if key != "rows":
            need(cached[key] == value, "Full original physical stage re-audit differs: " + name + "." + key)
    need(sha(experiment / name / "audit_source_snapshot" / ("audit_balanced_run_" + STAGE_SHA + ".py")) == STAGE_SHA,
         "Original executed stage auditor snapshot changed")
    return cached


def actual_case(row):
    result, assessment = row["result"], row["refine_assessment"]
    amount = result["received_ml"]
    return {"episode_id": row["episode_id"], "profile_key": row["profile_key"], "role": row["role"],
        "coarse_dose_tilt_deg_s": row["declared_policy"]["coarse_dose_tilt_deg_s"],
        "first_approach_tilt_deg": row["declared_policy"]["first_approach_tilt_deg"],
        "seed": row["seed"], "actual_received_ml": amount, "error_ml": amount - 40.,
        "absolute_error_ml": abs(amount - 40.), "physical_time_s": result["physical_time_s"],
        "all_label_peak_spill_particles": assessment["maximum_all_label_spilled_particles"],
        "valid_evidence": bool(assessment["valid_evidence"] and result["mass_conserved"] and result["all_states_finite"]),
        "primary_passed": assessment["feasible"]}


def phase_proof(experiment, name, report, cached):
    return {"stage": name, "protocol_id": report["protocol_id"], "audit_sha256": sha(experiment / name / "audit.json"),
        "auditor_source_sha256": STAGE_SHA, "input_artifact_sha256": report["input_artifact_sha256"],
        "completed_sidecar_sha256": {row["episode_id"]: sha(experiment / name / "completed" / (row["episode_id"] + ".json"))
                                    for row in report["rows"]},
        "raw_artifact_sha256": {relative: digest for row in report["rows"] for relative, digest in row["artifacts"].items()},
        "created_at_utc": cached["created_at_utc"], "actual_cases": [actual_case(row) for row in report["rows"]]}


def ranking(rows, stage):
    ranks = []
    for index, key in enumerate(stage.CANDIDATES):
        group = [row for row in rows if row["candidate_id"] == key]
        score = stage.score_group(group)
        rank = (2, index) if score is None else (*score, index)
        ranks.append((rank, key))
    return [{"profile_key": key, "rank": list(rank)} for rank, key in sorted(ranks)]


def common_checkpoint(experiment, model, calibration, cached, stage, root):
    chosen = stage.choose_candidate(calibration["rows"])
    need(model["task_kind"] == "fixed40ml_absolute05ml_24s_onegrain_v1"
         and model["experiment_plan_sha256"] == stage.PLAN_SHA
         and model["selected_profile_key"] == chosen
         and model["selected_first_approach_tilt_deg"] == stage.ANGLES[chosen]
         and model["fixed_coarse_dose_tilt_deg_s"] == 24.8 and model["retry_approach_tilt_deg"] == 87.2 and model["baseline_profile_key"] == "baseline"
         and model["baseline_first_approach_tilt_deg"] == 87.
         and model["both_roles_enable_retry"] is True and model["maximum_cycles"] == 2
         and model["only_first_angle_varies"] is True
         and model["only_coarse_dose_speed_varies"] is False
         and model["baseline_comparator_does_not_require_task_success"] is True
         and model["candidate_group_requires_all_joint_primary"] is True and model["fixed_parameters"] == {**stage.COARSE, **stage.MOTION},
         "Frozen profile changed more than the selected first approach angle")
    matrix = read(experiment / "calibration/matrix.json")
    need(model["profiles"] == calibration["profiles"] and model["profile_file"] == matrix["profile_file"]
         and model["profile_id"] == calibration["profile_id"]
         and model["profile_raw_sha256"] == calibration["profile_raw_sha256"]
         and model["calibration_proof"] == phase_proof(experiment, "calibration", calibration, cached)
         and model["known_baseline_equivalence"] == calibration["known_baseline_equivalence"]
         and model["auditor_source_sha256"] == STAGE_SHA
         and model["selector_source_sha256"] == sha(root / HERE / "select_and_freeze.py")
         and model["versions"] == calibration["versions"], "Calibration-only checkpoint provenance differs")
    for key in MAPS:
        need(model[key] == calibration[key], "Frozen current/source/input/old-history bindings changed: " + key)


def verify_selection(experiment, calibration, cached, stage, root):
    selection = read(experiment / "calibration_selection.json")
    need(selection["kind"] == "balanced40_calibration_selection_v1"
         and selection["calibration_gate_passed"] is True and calibration["calibration_gate_passed"] is True
         and selection["calibration_cases"] == 9 and selection["candidate_tie_order"] == list(stage.CANDIDATES)
         and selection["grouped_calibration_seeds"] == list(stage.KNOWN_SEEDS)
         and selection["feasible_group_rank_uses_mean_physical_time"] is True
         and selection["infeasible_group_rank_uses_mae_then_total_peak_spill"] is True
         and selection["known_case_is_seen_development"] is True
         and selection["validation_used_for_selection"] is False and selection["test_used_for_selection"] is False
         and selection["feasible_rank_uses_only_physical_time_then_exact_declared_order"] is True
         and selection["validation_may_gate_but_not_retune_or_fallback"] is True
         and selection["default_gui_deployment"] is False, "Selection used an unapproved phase or criterion")
    common_checkpoint(experiment, selection, calibration, cached, stage, root)
    need(selection["all_calibration_cases"] == [actual_case(row) for row in calibration["rows"]]
         and selection["candidate_scores"] == calibration["candidate_scores"]
         and selection["candidate_ranking"] == ranking(calibration["rows"], stage),
         "Known-case candidate ranking differs from independently recomputed all-record scores")
    need(stage.utc(selection["created_at_utc"]) > stage.utc(cached["created_at_utc"]),
         "Calibration choice froze before its completed independent audit")
    return selection


def checkpoint_meta(experiment, name, model):
    return {"file": name, "sha256": sha(experiment / name), "created_at_utc": model["created_at_utc"],
        "selected_profile_key": model["selected_profile_key"],
        "selected_first_approach_tilt_deg": model["selected_first_approach_tilt_deg"], "retry_approach_tilt_deg": 87.2}


def coarse_prefix_pair(baseline, candidate, run, ref):
    """First-angle treatment begins after the same measured quiet78 coarse anchor."""
    anchors = [row["status"].get("prefix_anchor_observation") for row in (baseline, candidate)]
    base = {"seed": candidate["seed"], "candidate_id": candidate["candidate_id"],
        "baseline_episode_id": baseline["episode_id"], "candidate_episode_id": candidate["episode_id"]}
    if any(anchor is None for anchor in anchors):
        return {**base, "verified": False, "endpoint_unavailable": True}
    endpoint = anchors[0]["physical_time_s"]
    records = [ref.gz(run / "traces" / (row["episode_id"] + ".steps.jsonl.gz"))[0] for row in (baseline, candidate)]
    steps = [[item for item in data if item["type"] == "step" and item["physical_time_s"] <= endpoint] for data in records]
    direct = [[item for item in data if item["type"] == "direct_step_evidence" and item["physical_time_s"] <= endpoint] for data in records]
    same = anchors[1]["physical_time_s"] == endpoint and len(steps[0]) == len(steps[1]) == len(direct[0]) == len(direct[1])
    for one, two in zip(*steps):
        same = same and all(one[key] == two[key] for key in ("step", "physical_time_s", "physics_dt_s", "source_time", "truth"))
        if one["step"]:
            same = same and one["solver_substeps"] == two["solver_substeps"]
        else:
            need(one["physical_time_s"] == two["physical_time_s"] == 0.
                 and min(one["solver_substeps"], two["solver_substeps"]) >= 4,
                 "Inherited substep-counter exception is limited to time0 before a completed frame")
    maximum_dense = []
    for rows in direct:
        times = [sample["time_s"] for item in rows for sample in item["kinematics"]["dense_240_hz_samples"]]
        need(all(time_s <= endpoint for time_s in times), "Quiet-coarse prefix cannot include future first-angle kinematics")
        maximum_dense.append(max(times) if times else None)
    for one, two in zip(*direct):
        same = same and all(one[key] == two[key] for key in ("step", "physical_time_s", "kinematics", "partitions", "maximum_all_label_spilled_particles"))
    return {**base, "verified": same, "endpoint_unavailable": False,
        "comparison_endpoint_kind": "actual_quiet_coarse78_before_first_angle_treatment",
        "coarse_anchor_physical_s": endpoint, "coarse_prefix_frames": len(steps[0]),
        "maximum_compared_dense_time_s": maximum_dense, "no_future_kinematics_included": True,
        "actual_coarse_water_motion_partitions_exactly_equal": same,
        "whole_first_cycle_equality_is_not_required": True,
        "zero_time_inherited_last_substeps": {"baseline": steps[0][0]["solver_substeps"],
            "candidate": steps[1][0]["solver_substeps"], "is_actual_completed_frame": False,
            "excluded_from_equivalence_only": True}}


def stop_receipt(experiment, names, gate):
    need(names in (["calibration"], ["calibration", "validation"]) and gate is False,
         "Only an actually failed completed development gate can stop before fresh tests")
    need(not (experiment / "test").exists() and not (experiment / "fixed_strategy.json").exists(),
         "Stopped development cannot hide fresh tests or a post-validation strategy")
    if names == ["calibration"]:
        need(not (experiment / "calibration_selection.json").exists(), "Failed calibration cannot have a frozen selected policy")
    outcome = "validation_gate_failed" if "validation" in names else "calibration_gate_failed"
    path = experiment / "finalization.json"
    receipt = read(path)
    need(receipt["kind"] == "balanced40_finalization_v1" and receipt["outcome"] == outcome
         and receipt["actual_stages"] == names and receipt["independent_test_cases"] == 0
         and receipt["no_retune_or_baseline_fallback"] is True, "Missing explicit bounded phase-stop receipt")
    return {"file": path.name, "sha256": sha(path), "outcome": outcome}


def audit(experiment, root=ROOT, *, development_stop=False):
    experiment, root = Path(experiment).resolve(), Path(root).resolve()
    stage = load_stage(root)
    plan = stage.validate_experiment_plan(experiment)
    names = actual_stages(experiment, development_stop)
    complete_gate(experiment, names, stage)
    calibration = stage.audit_stage(experiment / "calibration", "calibration", root=root, parent_plan_sha=stage.PLAN_SHA)
    cached = {"calibration": verify_cached(experiment, "calibration", calibration)}
    reports = {"calibration": calibration}
    selection = strategy = None
    if "validation" in names:
        selection = verify_selection(experiment, calibration, cached["calibration"], stage, root)
        chosen = selection["selected_profile_key"]
        reports["validation"] = stage.audit_stage(experiment / "validation", "validation", chosen, root=root, parent_plan_sha=stage.PLAN_SHA)
        cached["validation"] = verify_cached(experiment, "validation", reports["validation"])
        need(read(experiment / "validation/matrix.json")["calibration_selection_sha256"] == sha(experiment / "calibration_selection.json")
             and stage.utc(selection["created_at_utc"]) < min(stage.utc(row["actual_reset_started_utc"])
                 for row in reports["validation"]["episode_reports"]), "Choice did not freeze before actual fresh validation")
    for report in reports.values():
        for key in (*MAPS, "versions", "protected_manifest_raw_sha256", "profile_raw_sha256"):
            need(report[key] == calibration[key], "Static sources/inputs/old20/30 or selected profile changed between phases")
    for name, report in reports.items():
        need(stage.utc(cached[name]["created_at_utc"]) > max(stage.utc(row["completed_at_utc"]) for row in report["rows"]),
             "Independent phase audit did not follow all actually completed cases")
    finalization = None
    if development_stop:
        outcome = "validation_gate_failed" if "validation" in names else "calibration_gate_failed"
        gate = reports[names[-1]]["validation_gate_passed" if "validation" in names else "calibration_gate_passed"]
        need(gate is False and (selection is not None or not (experiment / "calibration_selection.json").exists()),
             "A development stop must preserve the actual failed gate without hidden selection")
        finalization = stop_receipt(experiment, names, gate)
    else:
        need(reports["validation"]["validation_gate_passed"] is True, "Failed validation may not start fresh tests")
        strategy = read(experiment / "fixed_strategy.json")
        need(strategy["kind"] == "balanced40_frozen_strategy_v1"
             and strategy["validation_gate_passed"] is True and strategy["candidate_not_retuned_or_replaced_by_validation"] is True
             and strategy["calibration_selection_sha256"] == sha(experiment / "calibration_selection.json")
             and strategy["calibration_selection"] == {"file": "calibration_selection.json",
                "sha256": sha(experiment / "calibration_selection.json"), "created_at_utc": selection["created_at_utc"]}
             and strategy["validation_proof"] == phase_proof(experiment, "validation", reports["validation"], cached["validation"])
             and strategy["calibration_cases"] == 9 and strategy["validation_cases"] == 4
             and strategy["test_used_for_selection"] is False and strategy["test_cases"] == 20
             and strategy["default_gui_deployment"] is False, "Post-validation strategy was retuned or lost its frozen choice")
        common_checkpoint(experiment, strategy, calibration, cached["calibration"], stage, root)
        need(strategy["selected_profile_key"] == selection["selected_profile_key"]
             and stage.utc(strategy["created_at_utc"]) > stage.utc(cached["validation"]["created_at_utc"]),
             "Final strategy did not freeze after completed validation")
        reports["test"] = stage.audit_stage(experiment / "test", "test", selection["selected_profile_key"], root=root, parent_plan_sha=stage.PLAN_SHA)
        cached["test"] = verify_cached(experiment, "test", reports["test"])
        need(stage.utc(cached["test"]["created_at_utc"]) > max(stage.utc(row["completed_at_utc"]) for row in reports["test"]["rows"]),
             "Final test audit preceded a completed actual case")
        for key in (*MAPS, "versions", "protected_manifest_raw_sha256", "profile_raw_sha256"):
            need(reports["test"][key] == calibration[key], "A test changed frozen source/input/GUI/old-history bindings")
        matrix = read(experiment / "test/matrix.json")
        need(matrix["calibration_selection_sha256"] == sha(experiment / "calibration_selection.json")
             and matrix["fixed_strategy_sha256"] == sha(experiment / "fixed_strategy.json")
             and stage.utc(strategy["created_at_utc"]) < min(stage.utc(row["actual_reset_started_utc"])
                 for row in reports["test"]["episode_reports"]), "Strategy did not freeze before every actual test reset")
        outcome = "fresh_test_completed"
    all_reports = [row for name in names for row in reports[name]["episode_reports"]]
    total = sum(stage.COUNTS[name] for name in names)
    need(len(all_reports) == len({row["episode_id"] for row in all_reports}) == total, "A physical attempt was removed or duplicated")
    physical, _ = stage.load_physical_audit(root)
    ref = physical.load_cpu_reference(root)
    pairs = []
    for name in names:
        rows = reports[name]["rows"]
        for candidate in (row for row in rows if row["role"] == "candidate"):
            baseline = next(row for row in rows if row["role"] == "baseline" and row["seed"] == candidate["seed"])
            pair = coarse_prefix_pair(baseline, candidate, experiment / name, ref)
            pair["stage"] = name
            pairs.append(pair)
    return {"complete": True, "pass": True, "kind": stage.KIND, "scope": stage.SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan_sha256": stage.PLAN_SHA,
        "outcome": outcome, "development_stop": development_stop, "actual_stages": names,
        "budgets": {"calibration": 9, "validation": 4 if "validation" in names else 0,
            "test": 20 if "test" in names else 0, "total": total, "maximum_total": 33},
        "calibration_gate_passed": calibration["calibration_gate_passed"],
        "validation_gate_passed": reports["validation"]["validation_gate_passed"] if "validation" in names else None,
        "selected_profile_key": calibration["selected_profile_key"],
        "calibration_selection": checkpoint_meta(experiment, "calibration_selection.json", selection) if selection else None,
        "fixed_strategy": checkpoint_meta(experiment, "fixed_strategy.json", strategy) if strategy else None,
        "finalization": finalization, "known_baseline_equivalence": calibration["known_baseline_equivalence"],
        "balanced_first_angle_parameter_calibration_performed": True, "calibration_grid_selection_performed": True,
        "collector_fits_a_policy_model": False, "validation_used_for_retuning_or_fallback": False,
        "test_used_for_selection": False, "only_first_angle_varies": True, "fixed_coarse_dose_tilt_deg_s": 24.8, "fixed_retry_angle_deg": 87.2,
        "protected_file_sha256": calibration["protected_file_sha256"],
        "protected_manifest_raw_sha256": calibration["protected_manifest_raw_sha256"],
        "old20_and30_versions_unchanged": True, "first40_versions_unchanged": True, "coarse40_versions_unchanged": True,
        "baseline_comparator_does_not_require_task_success": True,
        "candidate_group_requires_all_joint_primary": True,
        "all_physical_sources_inputs_gui_unchanged": True,
        "coarse_prefix_equivalence": pairs, "all_coarse_prefix_physical_states_equal": all(pair["verified"] for pair in pairs),
        "stage_artifact_sha256": {name: {**report["input_artifact_sha256"], "audit.json": sha(experiment / name / "audit.json")}
            for name, report in reports.items()},
        "stage_reports": {name: {key: value for key, value in report.items() if key != "rows"} for name, report in reports.items()},
        "episode_reports": all_reports, "original_strict_task_was_not_relaxed_or_relabelled": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "auditor_source_file": HERE + "audit_balanced_experiment.py", "auditor_source_sha256": sha(__file__),
        "pinned_stage_audit_sha256": STAGE_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--experiment", dest="experiment", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-stop", action="store_true")
    args = parser.parse_args()
    result = audit(args.experiment, args.workspace, development_stop=args.development_stop)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("episode_reports", "stage_reports")}, indent=2))


if __name__ == "__main__":
    main()
