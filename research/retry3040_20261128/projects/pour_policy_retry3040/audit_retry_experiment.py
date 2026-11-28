"""Independent final audit of bounded30/40mL development and optional fresh tests.

This records evidence integrity and a predeclared paired experiment. It does
not fit a model, choose a winner using tests, or certify a real robot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_retry3040/"
STAGE_SHA = "f658538d89e404eaef4b9ba83d813c64e19d375983dfa0bdcf0fc61c4c1a1d75"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def load_stage(root=ROOT):
    path = root / HERE / "audit_retry_run.py"
    need(sha(path) == STAGE_SHA, "Frozen pilot/stage auditor changed")
    spec = importlib.util.spec_from_file_location("retry3040_pinned_stage_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def development_stages(experiment):
    names = [name for name in ("pilot_001", "pilot_002") if (experiment / name).is_dir()]
    need(names and names[0] == "pilot_001", "The first bounded development stage must be retained")
    extras = [path.name for path in experiment.glob("pilot_*") if path.is_dir() and path.name not in names]
    need(not extras, "Undeclared extra development attempts exist")
    return names


def complete_gate(experiment, stage, names):
    """Do not open raw journals or profiles until every declared stage is complete."""
    for name in names:
        run = experiment / name
        need((run / "summary.json").is_file() and (run / "job_state.json").is_file(),
             "All actual development and required fresh-test cases must complete before final audit")
        summary, job = read(run / "summary.json"), read(run / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == stage.COUNTS[name]
             and summary.get("stage") == name and job.get("state") == "complete"
             and job.get("pid") is None and not (run / "worker.lock").exists(),
             "All actual development and required fresh-test cases must complete before final audit")


def cached_audit(experiment, name, report, stage):
    cached = read(experiment / name / "audit.json")
    need(cached["complete"] is True and cached["pass"] is True and cached["stage"] == name
         and cached["auditor_source_sha256"] == STAGE_SHA, "Cached stage ownership/source differs")
    for key, value in report.items():
        if key != "rows":
            need(cached[key] == value, "Independent raw stage re-audit differs: " + name + "." + key)
    snapshot = experiment / name / "audit_source_snapshot" / ("audit_retry_run_" + STAGE_SHA + ".py")
    need(sha(snapshot) == STAGE_SHA, "Original stage audit snapshot changed")
    return cached


def verify_freeze_times(end_pilot, pilot_audited, strategy_created, first_actual_test):
    need(end_pilot < pilot_audited < strategy_created < first_actual_test,
         "Pilot/audit/fixed policy were not completed and frozen before the actual fresh-test reset")


def adjustment_profiles(first):
    """Only one signed0.1degree correction, based on completed development doses."""
    need(first["all_non_dose_safety_passed"] is True and first["development_gate_passed"] is False,
         "A second engineering stage cannot hide unsafe evidence or retune passing profiles")
    output = {goal: dict(profile) for goal, profile in first["goal_profiles"].items()}
    failed = False
    for goal in (30., 40.):
        candidate = [row for row in first["episode_reports"] if row["target_ml"] == goal and row["role"] == "candidate"]
        need(len(candidate) == 1, "Complete paired candidate development goals required")
        row = candidate[0]
        if not row["refine_assessment"]["dose_passed"]:
            failed = True
            shift = -.1 if row["error_ml"] > 0. else .1
            output[format(goal, "g")] = {key: round(value + shift, 10)
                                        for key, value in output[format(goal, "g")].items()}
    need(failed, "No failed goal permits an extra angle calibration stage")
    return output


def paired_first_cycle(base, candidate, run, ref):
    """Compare actual first-cycle physics, excluding only inherited t0 counter."""
    def endpoint(row):
        return next((item for item in row["response_events"]
                     if item["type"] in ("response_cycle_complete", "response_cycle_unsettled")), None)
    ends = endpoint(base), endpoint(candidate)
    if all(item is None for item in ends) and all(not any(event["type"] == "response_cycle_start"
            for event in row["response_events"]) for row in (base, candidate)):
        anchors = base["status"].get("prefix_anchor_observation"), candidate["status"].get("prefix_anchor_observation")
        if any(item is None for item in anchors):
            return {"target_ml": base["target_ml"], "seed": base["seed"], "verified": False, "endpoint_unavailable": True}
        first = anchors[0]["physical_time_s"]
        events_equal = first == anchors[1]["physical_time_s"] and base["response_events"] == candidate["response_events"]
        endpoint_kind = "actual_quiet_coarse_anchor_no_cycle_needed"
    elif any(item is None for item in ends):
        return {"target_ml": base["target_ml"], "seed": base["seed"], "verified": False, "endpoint_unavailable": True}
    else:
        events_equal = (ends[0] == ends[1] and base["response_events"][:ends[0]["event_id"] + 1]
                        == candidate["response_events"][:ends[1]["event_id"] + 1])
        first = ends[0]["physical_time_s"]
        endpoint_kind = "actual_first_complete_or_unsettled_whole_cycle"
    records = [ref.gz(run / "traces" / (row["episode_id"] + ".steps.jsonl.gz"))[0] for row in (base, candidate)]
    steps = [[item for item in items if item["type"] == "step" and item["physical_time_s"] <= first]
             for items in records]
    direct = [[item for item in items if item["type"] == "direct_step_evidence" and item["physical_time_s"] <= first]
              for items in records]
    equivalent = len(steps[0]) == len(steps[1]) == len(direct[0]) == len(direct[1])
    for one, two in zip(*steps):
        equivalent = equivalent and all(one[key] == two[key] for key in
            ("step", "physical_time_s", "physics_dt_s", "source_time", "truth"))
        if one["step"] != 0:
            equivalent = equivalent and one["solver_substeps"] == two["solver_substeps"]
        else:
            need(one["physical_time_s"] == two["physical_time_s"] == 0.
                 and one["truth"]["received_ml"] == two["truth"]["received_ml"] == 0.
                 and min(one["solver_substeps"], two["solver_substeps"]) >= 4,
                 "The pairing counter exception applies only to zero physical time")
    for one, two in zip(*direct):
        equivalent = equivalent and all(one[key] == two[key] for key in
            ("step", "physical_time_s", "kinematics", "partitions", "maximum_all_label_spilled_particles"))
    return {"target_ml": base["target_ml"], "seed": base["seed"], "baseline_episode_id": base["episode_id"],
        "candidate_episode_id": candidate["episode_id"], "first_cycle_end_physical_s": first,
        "comparison_endpoint_kind": endpoint_kind,
        "first_cycle_frames": len(steps[0]), "original_first_cycle_events_exactly_equal": events_equal,
        "actual_water_motion_partitions_exactly_equal": equivalent, "verified": events_equal and equivalent,
        "zero_time_inherited_last_substeps": {"baseline": steps[0][0]["solver_substeps"],
            "candidate": steps[1][0]["solver_substeps"], "is_actual_completed_frame": False,
            "excluded_from_equivalence_only": True}}


def core_inputs(report):
    return {name: digest for name, digest in report["input_sha256"].items() if not name.startswith(HERE)}


def validate_adjustment(experiment, first, second):
    expected = adjustment_profiles(first)
    need(second["goal_profiles"] == expected, "Optional second stage exceeded its predeclared signed0.1degree correction")
    receipt_file = experiment / "development_adjustment_001.json"
    receipt = read(receipt_file)
    changes = [{"target_ml": row["target_ml"], "actual_prior_error_ml": row["error_ml"],
                "delta_degrees": -.1 if row["error_ml"] > 0. else .1}
               for row in first["episode_reports"] if row["role"] == "candidate"
               and not row["refine_assessment"]["dose_passed"]]
    need(receipt["kind"] == "retry3040_preregistered_development_adjustment_v1"
         and receipt["pilot001_audit_sha256"] == sha(experiment / "pilot_001/audit.json")
         and receipt["changes"] == changes
         and receipt["old_profile_sha256"] == first["goal_profile_raw_sha256"]
         and receipt["new_profile_sha256"] == second["goal_profile_raw_sha256"]
         and receipt["no_code_or_speed_or_criterion_change"] is True
         and receipt["test_data_read_or_used"] is False, "Development calibration receipt differs from actual signed dose errors")
    return {"file": receipt_file.name, "sha256": sha(receipt_file), "changes": changes}


def stop_receipt(experiment, development, gate_passed):
    need(gate_passed is False, "Passing development cannot be relabelled as a failed gate")
    need(not (experiment / "test").exists() and not (experiment / "fixed_strategy.json").exists(),
         "A finalized development stop cannot hide tests or a frozen strategy")
    receipt_file = experiment / "finalization.json"
    receipt = read(receipt_file)
    need(receipt["kind"] == "retry3040_finalization_v1" and receipt["outcome"] == "development_gate_failed"
         and receipt["actual_development_stages"] == development and receipt["independent_test_cases"] == 0
         and receipt["bounds_exhausted_or_unsafe"] is True, "No explicit bounded development-stop receipt")
    return {"file": receipt_file.name, "sha256": sha(receipt_file), "outcome": receipt["outcome"]}


def verify_strategy(experiment, development, reports, cached, plan, stage, root):
    strategy_file = experiment / "fixed_strategy.json"
    strategy = read(strategy_file)
    latest = reports[development[-1]]
    need(strategy["kind"] == "fixed3040_conditional_second_cycle_strategy_v1"
         and strategy["task_kind"] == "fixed30_40ml_absolute05ml_24s_onegrain_v1"
         and strategy["experiment_plan_sha256"] == stage.PLAN_SHA
         and strategy["completed_development_stages"] == development
         and strategy["goal_profiles"] == latest["goal_profiles"]
         and strategy["goal_profile_raw_sha256"] == latest["goal_profile_raw_sha256"]
         and strategy["development_gate_passed"] is True
         and strategy["targets_ml"] == list(stage.TARGETS)
         and strategy["selected_development_stage"] == development[-1]
         and strategy["development_cases"] == 4 * len(development)
         and strategy["maximum_cycles_baseline"] == 1 and strategy["maximum_cycles_candidate"] == 2
         and strategy["test_cases"] == 40 and strategy["test_goal_seed_conditions"] == 20
         and strategy["no_new_training_search_or_neural_policy"] is True
         and strategy["test_used_for_selection"] is False and strategy["default_gui_deployment"] is False,
         "Strategy does not freeze the completed passing engineering profile before tests")
    for key in ("source_sha256", "input_sha256", "gui_sha256"):
        need(strategy[key] == latest[key], "Frozen source/input/GUI differs from the passing development implementation")
    need(strategy["fixed_parameters"] == {**stage.COARSE, **stage.MOTION}
         and strategy["versions"] == latest["versions"]
         and strategy["stage_auditor_source_sha256"] == STAGE_SHA,
         "Fixed motion/actual runtime or reviewed stage auditor changed after calibration")
    bindings = {name: {**report["input_artifact_sha256"], "audit.json": sha(experiment / name / "audit.json")}
                for name, report in reports.items() if name in development}
    need(strategy["development_artifact_sha256"] == bindings
         and strategy["freeze_helper_sha256"] == sha(root / HERE / "freeze_strategy.py"),
         "Frozen policy lost a failed development stage or its exact audit implementation")
    records = []
    for name in development:
        report = reports[name]
        matrix = read(experiment / name / "matrix.json")
        candidates = {}
        for target in stage.TARGETS:
            row = next(row for row in report["episode_reports"] if row["role"] == "candidate" and row["target_ml"] == target)
            candidates[format(target, "g")] = {"target_ml": target, "actual_received_ml": row["received_ml"],
                "error_ml": row["received_ml"] - target, "primary_passed": row["refine_assessment"]["dose_passed"]}
        raw = {relative: digest for row in report["rows"] for relative, digest in row["artifacts"].items()}
        records.append({"stage": name, "protocol_id": report["protocol_id"],
            "audit_sha256": sha(experiment / name / "audit.json"), "auditor_source_sha256": STAGE_SHA,
            "input_artifact_sha256": report["input_artifact_sha256"],
            "completed_sidecar_sha256": {row["episode_id"]: sha(experiment / name / "completed" / (row["episode_id"] + ".json"))
                                        for row in report["rows"]},
            "raw_artifact_sha256": raw, "goal_profile_file": matrix["goal_profile_file"],
            "goal_profile_id": report["goal_profile_id"], "goal_profile_raw_sha256": report["goal_profile_raw_sha256"],
            "goal_profiles": report["goal_profiles"], "candidate_by_goal": candidates,
            "all_non_dose_safety_passed": report["all_non_dose_safety_passed"],
            "development_gate_passed": report["development_gate_passed"]})
    need(strategy["development_stages"] == records, "Frozen development ledger lost a failed case/raw trace/profile")
    selected_matrix = read(experiment / development[-1] / "matrix.json")
    need(strategy["goal_profile_file"] == selected_matrix["goal_profile_file"]
         and strategy["goal_profile_id"] == latest["goal_profile_id"], "Selected external profile identity changed")
    adjustment = None if len(development) == 1 else {"file": "development_adjustment_001.json",
        "sha256": sha(experiment / "development_adjustment_001.json")}
    need(strategy["development_adjustment"] == adjustment, "Frozen strategy lost the prescribed adjustment receipt")
    stage.utc(strategy["created_at_utc"])
    return strategy


def audit(experiment, root=ROOT, *, development_stop=False):
    experiment, root = Path(experiment).resolve(), Path(root).resolve()
    stage = load_stage(root)
    plan = stage.validate_experiment_plan(experiment)
    development = development_stages(experiment)
    names = [*development, *([] if development_stop else ["test"])]
    if development_stop:
        need(not (experiment / "test").exists() and not (experiment / "fixed_strategy.json").exists(),
             "A finalized development stop cannot hide already-started tests or a strategy checkpoint")
    complete_gate(experiment, stage, names)
    reports, cached = {}, {}
    for name in names:
        reports[name] = stage.audit_stage(experiment / name, name, root=root, parent_plan_sha=stage.PLAN_SHA)
        cached[name] = cached_audit(experiment, name, reports[name], stage)
    first, latest = reports[development[0]], reports[development[-1]]
    for name, report in reports.items():
        need(report["source_sha256"] == first["source_sha256"] and core_inputs(report) == core_inputs(first)
             and len(core_inputs(report)) == 6 and report["gui_sha256"] == first["gui_sha256"]
             and report["versions"] == first["versions"], "Core physics/code/cache/GUI/runtime changed between stages")
    initial = read(root / plan["initial_goal_profile_file"])
    need(first["goal_profiles"] == initial["goal_profiles"]
         and first["goal_profile_raw_sha256"] == plan["initial_goal_profile_raw_sha256"],
         "The first actual engineering stage no longer uses the preregistered proposal")
    adjustment = None
    if len(development) == 2:
        second = reports["pilot_002"]
        adjustment = validate_adjustment(experiment, first, second)
        need(max(stage.utc(row["completed_at_utc"]) for row in first["rows"])
             < stage.utc(cached["pilot_001"]["created_at_utc"])
             < min(stage.utc(row["actual_reset_started_utc"]) for row in second["episode_reports"]),
             "Optional second engineering stage started before the first was independently audited")
    finalization, strategy_meta = None, None
    test_roles = {}
    if development_stop:
        finalization = stop_receipt(experiment, development, latest["development_gate_passed"])
        outcome = "development_gate_failed"
    else:
        need(latest["development_gate_passed"] is True, "Failed engineering gate cannot authorize fresh tests")
        strategy = verify_strategy(experiment, development, reports, cached, plan, stage, root)
        matrix = read(experiment / "test/matrix.json")
        need(matrix["fixed_strategy_sha256"] == sha(experiment / "fixed_strategy.json")
             and matrix["goal_profiles"] == latest["goal_profiles"] == reports["test"]["goal_profiles"]
             and reports["test"]["input_sha256"] == latest["input_sha256"],
             "Fresh-test matrix/profile/input does not bind the same frozen strategy")
        verify_freeze_times(max(stage.utc(row["completed_at_utc"]) for row in latest["rows"]),
            max(stage.utc(cached[name]["created_at_utc"]) for name in development),
            stage.utc(strategy["created_at_utc"]),
            min(stage.utc(row["actual_reset_started_utc"]) for row in reports["test"]["episode_reports"]))
        strategy_meta = {"file": "fixed_strategy.json", "sha256": sha(experiment / "fixed_strategy.json"),
            "created_at_utc": strategy["created_at_utc"], "goal_profiles": strategy["goal_profiles"],
            "goal_profile_raw_sha256": strategy["goal_profile_raw_sha256"]}
        for target in stage.TARGETS:
            goal_roles = {}
            for role in ("baseline", "candidate"):
                group = [row for row in reports["test"]["episode_reports"]
                         if row["target_ml"] == target and row["role"] == role]
                need(len(group) == 10 and {row["seed"] for row in group} == set(stage.TEST),
                     "Every goal/role requires the complete ten fresh conditions")
                goal_roles[role] = {"cases": 10,
                    "primary_passes": sum(row["refine_assessment"]["feasible"] for row in group),
                    "easy_passes": sum(row["easy_assessment"]["feasible"] for row in group),
                    "original_strict_passes": sum(row["strict_assessment"]["feasible"] for row in group),
                    "all_primary_passed": all(row["refine_assessment"]["feasible"] for row in group)}
            test_roles[format(target, "g")] = goal_roles
        outcome = "fresh_test_completed"
    all_reports = [row for name in names for row in reports[name]["episode_reports"]]
    total = 4 * len(development) + (0 if development_stop else 40)
    need(len(all_reports) == len({row["episode_id"] for row in all_reports}) == total,
         "Complete actual development/test attempts were lost or duplicated")
    physical, _ = stage.load_physical_audit(root)
    reference = physical.load_cpu_reference(root)
    pairs = []
    for name in names:
        seeds = stage.TEST if name == "test" else (stage.PILOT_SEED,)
        for target in stage.TARGETS:
            for seed in seeds:
                baseline = next(row for row in reports[name]["rows"] if row["target_ml"] == target
                                and row["seed"] == seed and row["role"] == "baseline")
                candidate = next(row for row in reports[name]["rows"] if row["target_ml"] == target
                                 and row["seed"] == seed and row["role"] == "candidate")
                pair = paired_first_cycle(baseline, candidate, experiment / name, reference)
                pair["stage"] = name
                pairs.append(pair)
    return {"complete": True, "pass": True, "kind": stage.KIND, "scope": stage.SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan_sha256": stage.PLAN_SHA,
        "outcome": outcome, "development_stop": development_stop, "actual_development_stages": development,
        "budgets": {"development": 4 * len(development), "test": 0 if development_stop else 40, "total": total,
            "maximum_development": 8, "planned_independent_test": 40},
        "independent_test_not_run": development_stop, "independent_test_conditions": 0 if development_stop else 20,
        "development_gate_passed": latest["development_gate_passed"], "fixed_strategy": strategy_meta,
        "finalization": finalization, "development_adjustment": adjustment, "test_role_results": test_roles,
        "fixed_policy_not_selected_from_test": True, "test_used_for_selection": False,
        "no_new_training_search_or_neural_policy": True,
        "all_physical_sources_geometry_inputs_gui_unchanged": True,
        "only_declared_goal_profile_inputs_may_differ": True,
        "paired_first_cycle_equivalence": pairs,
        "all_paired_first_cycle_physical_states_equal": all(pair["verified"] for pair in pairs),
        "stage_artifact_sha256": {name: {**report["input_artifact_sha256"], "audit.json": sha(experiment / name / "audit.json")}
            for name, report in reports.items()},
        "stage_reports": {name: {key: value for key, value in report.items() if key != "rows"}
            for name, report in reports.items()}, "episode_reports": all_reports,
        "original_strict_task_was_not_relaxed_or_relabelled": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "auditor_source_file": HERE + "audit_retry_experiment.py", "auditor_source_sha256": sha(__file__),
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
    print(json.dumps({key: value for key, value in result.items() if key not in ("stage_reports", "episode_reports")}, indent=2))


if __name__ == "__main__":
    main()
