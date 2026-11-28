"""Independent final audit of one seen pilot and twenty new fixed-policy tests.

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
HERE = "projects/pour_policy_retry20/"
STAGE_SHA = "f10e0d87cb2e56d18af7c727a4967e25090d8e4ef3bb06fcb966c2ddf8a916fc"


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
    spec = importlib.util.spec_from_file_location("retry20_pinned_stage_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_gate(experiment, stage):
    """Do not open fixed policy or fresh-test journals before all21 are complete."""
    for name, count in stage.COUNTS.items():
        run = experiment / name
        need((run / "summary.json").is_file() and (run / "job_state.json").is_file(),
             "All1+20 cases must complete before final audit")
        summary, job = read(run / "summary.json"), read(run / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and summary.get("stage") == name and job.get("state") == "complete"
             and job.get("pid") is None and not (run / "worker.lock").exists(),
             "All1+20 cases must complete before final audit")


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


def strategy_checkpoint(experiment, pilot, cached, plan, stage, root):
    path = experiment / "fixed_strategy.json"
    model = read(path)
    need(model["kind"] == "fixed20_conditional_second_cycle_strategy_v1"
         and model["task_kind"] == "fixed20ml_absolute05ml_24s_onegrain_v1"
         and model["experiment_plan_sha256"] == stage.PLAN_SHA
         and model["profiles"] == plan["profiles"]
         and model["no_data_fitting_or_model_selection"] is True and model["test_used_for_selection"] is False
         and model["default_gui_deployment"] is False
         and model["pilot_uses_seen_previous_test_seed"] is True and model["pilot_joint_task_passed"] is True,
         "Fixed declared strategy became a fitted/test-selected or implicitly deployed model")
    expected = {**stage.COARSE, **stage.MOTION, "first_approach_tilt_deg": 84.,
        "retry_approach_tilt_deg": 84.1, "maximum_cycles_baseline": 1, "maximum_cycles_candidate": 2,
        "target_ml": 20., "absolute_tolerance_ml": .5, "deadline_s": 24., "peak_all_label_spill_limit": 1}
    need(model["fixed_parameters"] == expected, "Fixed first/retry motion or task parameters changed after pilot")
    for key in ("source_sha256", "input_sha256", "gui_sha256"):
        need(model[key] == pilot[key], "Frozen strategy changes original physical provenance: " + key)
    need(model["pilot_protocol_id"] == pilot["protocol_id"]
         and model["pilot_journal_sha256"] == pilot["input_artifact_sha256"]["episodes.jsonl"]
         and model["pilot_audit_sha256"] == sha(experiment / "pilot/audit.json")
         and model["freeze_helper_sha256"] == sha(root / HERE / "freeze_strategy.py")
         and cached["pilot_gate_passed"] is True,
         "Fixed strategy does not bind the actual completed/independently audited seen-case gate")
    stage.utc(model["created_at_utc"])
    return model


def paired_first_cycle(base, candidate, run, ref):
    """Compare actual first-cycle physics, excluding only inherited t0 counter."""
    def endpoint(row):
        return next((item for item in row["response_events"]
                     if item["type"] in ("response_cycle_complete", "response_cycle_unsettled")), None)
    ends = endpoint(base), endpoint(candidate)
    if any(item is None for item in ends):
        return {"seed": base["seed"], "verified": False, "endpoint_unavailable": True}
    events_equal = (ends[0] == ends[1] and base["response_events"][:ends[0]["event_id"] + 1]
                    == candidate["response_events"][:ends[1]["event_id"] + 1])
    first = ends[0]["physical_time_s"]
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
    return {"seed": base["seed"], "baseline_episode_id": base["episode_id"],
        "candidate_episode_id": candidate["episode_id"], "first_cycle_end_physical_s": first,
        "first_cycle_frames": len(steps[0]), "original_first_cycle_events_exactly_equal": events_equal,
        "actual_water_motion_partitions_exactly_equal": equivalent, "verified": events_equal and equivalent,
        "zero_time_inherited_last_substeps": {"baseline": steps[0][0]["solver_substeps"],
            "candidate": steps[1][0]["solver_substeps"], "is_actual_completed_frame": False,
            "excluded_from_equivalence_only": True}}


def audit(experiment, root=ROOT):
    experiment, root = Path(experiment).resolve(), Path(root).resolve()
    stage = load_stage(root)
    plan = stage.validate_experiment_plan(experiment)
    complete_gate(experiment, stage)
    pilot = stage.audit_stage(experiment / "pilot", "pilot", root=root, parent_plan_sha=stage.PLAN_SHA)
    cached_pilot = cached_audit(experiment, "pilot", pilot, stage)
    need(pilot["pilot_gate_passed"] is True, "Seen-case pilot gate did not authorize fresh tests")
    strategy = strategy_checkpoint(experiment, pilot, cached_pilot, plan, stage, root)
    test = stage.audit_stage(experiment / "test", "test", root=root, parent_plan_sha=stage.PLAN_SHA)
    cached_audit(experiment, "test", test, stage)
    for key in ("source_sha256", "input_sha256", "gui_sha256", "versions"):
        need(test[key] == pilot[key], "Physical implementation/assets/GUI/versions changed after fixed policy freeze")
    matrix = read(experiment / "test/matrix.json")
    need(matrix["fixed_strategy_sha256"] == sha(experiment / "fixed_strategy.json"),
         "Fresh-test matrix did not freeze the same predeclared fixed strategy")
    verify_freeze_times(max(stage.utc(row["completed_at_utc"]) for row in pilot["rows"]),
        stage.utc(cached_pilot["created_at_utc"]), stage.utc(strategy["created_at_utc"]),
        min(stage.utc(row["actual_reset_started_utc"]) for row in test["episode_reports"]))
    reports = [*pilot["episode_reports"], *test["episode_reports"]]
    need(len(reports) == len({row["episode_id"] for row in reports}) == 21
         and {row["seed"] for row in test["episode_reports"]} == set(stage.TEST)
         and stage.PILOT_SEED not in stage.TEST, "Seen pilot leaked into the new held-out paired test")
    test_roles = {}
    for role in ("baseline", "candidate"):
        group = [row for row in test["episode_reports"] if row["role"] == role]
        need(len(group) == 10 and {row["seed"] for row in group} == set(stage.TEST),
             "Fixed role lacks all ten independent paired conditions")
        test_roles[role] = {"cases": 10, "primary_passes": sum(row["refine_assessment"]["feasible"] for row in group),
            "easy_passes": sum(row["easy_assessment"]["feasible"] for row in group),
            "original_strict_passes": sum(row["strict_assessment"]["feasible"] for row in group),
            "all_primary_passed": all(row["refine_assessment"]["feasible"] for row in group)}
    physical, _ = stage.load_physical_audit(root)
    reference = physical.load_cpu_reference(root)
    pairs = [paired_first_cycle(next(row for row in test["rows"] if row["role"] == "baseline" and row["seed"] == seed),
             next(row for row in test["rows"] if row["role"] == "candidate" and row["seed"] == seed),
             experiment / "test", reference) for seed in stage.TEST]
    stages = {"pilot": pilot, "test": test}
    return {"complete": True, "pass": True, "kind": stage.KIND, "scope": stage.SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan_sha256": stage.PLAN_SHA,
        "budgets": {**stage.COUNTS, "total": 21}, "pilot_gate_passed": True,
        "pilot_uses_known_previous_test_seed": True, "pilot_is_not_independent_validation": True,
        "fixed_strategy": {"file": "fixed_strategy.json", "sha256": sha(experiment / "fixed_strategy.json"),
            "created_at_utc": strategy["created_at_utc"], "profiles": strategy["profiles"],
            "fixed_parameters": strategy["fixed_parameters"]},
        "fixed_policy_not_selected_from_test": True, "no_data_fitting_or_model_selection": True,
        "test_used_for_selection": False, "fixed_strategy_frozen_before_actual_test_reset": True,
        "test_role_results": test_roles, "all_physical_sources_inputs_gui_unchanged": True,
        "paired_first_cycle_equivalence": pairs,
        "all_paired_first_cycle_physical_states_equal": all(pair["verified"] for pair in pairs),
        "stage_artifact_sha256": {name: {**report["input_artifact_sha256"], "audit.json": sha(experiment / name / "audit.json")}
            for name, report in stages.items()},
        "stage_reports": {name: {key: value for key, value in report.items() if key != "rows"}
            for name, report in stages.items()}, "episode_reports": reports,
        "original_strict_task_was_not_relaxed_or_relabelled": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "auditor_source_file": HERE + "audit_retry_experiment.py", "auditor_source_sha256": sha(__file__),
        "pinned_stage_audit_sha256": STAGE_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--experiment", dest="experiment", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.workspace)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("stage_reports", "episode_reports")}, indent=2))


if __name__ == "__main__":
    main()
